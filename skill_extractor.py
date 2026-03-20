# skill_extractor.py
# ─────────────────────────────────────────────────────────────────────────────
# Fine-tunes dslim/bert-base-NER on Person B's real data.
#
# Data sources (from Person B):
#   data/Resume.csv   — 2484 resumes, column: "Resume_str" or last text column
#   data/Skills.xlsx  — 62580 O*NET skills, column: "Element Name"
#
# Label set (Phase 1): O, B-SKILL, I-SKILL
# Phase 2 will add EXPERIENCE + EDUCATION once skill extraction is solid.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import classification_report, f1_score

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL   = "dslim/bert-base-NER"
OUTPUT_DIR   = "models/skill_extractor"
RESUME_FILE  = "data/Resume.csv"
SKILLS_FILE  = "data/Skills.xlsx"
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
VAL_SPLIT    = 0.1

# ── Label schema (Phase 1 — skills only) ─────────────────────────────────────
# Phase 2 will extend to: B-EXPERIENCE I-EXPERIENCE B-EDUCATION I-EDUCATION

LABELS   = ["O", "B-SKILL", "I-SKILL"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ── Skill vocabulary loader ───────────────────────────────────────────────────

def load_skill_vocab(skills_file: str = SKILLS_FILE) -> set:
    """
    Loads the O*NET skills vocabulary from Person B's Skills.xlsx.
    Returns a set of lowercase skill strings (single words + bigrams).
    """
    df = pd.read_excel(skills_file)

    # Find the skill name column — tries known names then falls back
    # Skills.xlsx has 15 columns — we want "Element Name" (index 3)
    # Match exactly to avoid picking up short columns like "Scale Name"
    skill_col = None
    for col in df.columns:
        if col.strip().lower() == "element name":
            skill_col = col
            break
    if skill_col is None:
        # fallback: pick column with most unique string values
        str_cols = df.select_dtypes(include="object").columns.tolist()
        skill_col = max(str_cols, key=lambda c: df[c].nunique())
        print(f"[WARN] 'Element Name' not found — using '{skill_col}' ({df[skill_col].nunique()} unique values)")

    vocab = set(df[skill_col].dropna().str.lower().str.strip().tolist())
    print(f"[INFO] Skill vocabulary: {len(vocab)} terms from {skills_file}")
    return vocab


# ── Vocabulary-based BIO tagger ───────────────────────────────────────────────

def tag_resume(text: str, skill_vocab: set) -> tuple[list, list]:
    """
    Tags each word in resume text as B-SKILL, I-SKILL, or O
    using the O*NET skill vocabulary. Tries bigrams first.
    Returns (words, labels).
    """
    words = text.split()
    labels = []
    i = 0
    while i < len(words):
        clean = words[i].lower().strip(".,;:()[]\"'")

        # Try bigram (e.g. "machine learning", "data analysis")
        if i + 1 < len(words):
            clean_next = words[i+1].lower().strip(".,;:()[]\"'")
            bigram = clean + " " + clean_next
            if bigram in skill_vocab:
                labels.append("B-SKILL")
                labels.append("I-SKILL")
                i += 2
                continue

        # Single word
        labels.append("B-SKILL" if clean in skill_vocab else "O")
        i += 1

    return words, labels


# ── Dataset ───────────────────────────────────────────────────────────────────

class ResumeNERDataset(Dataset):
    """
    Accepts a list of raw resume strings + the O*NET skill vocabulary.
    Tags on-the-fly using tag_resume() — no pre-labelled .jsonl needed.
    """
    def __init__(self, texts: list, tokenizer, skill_vocab: set, max_len: int = MAX_LEN):
        self.examples   = []
        self.tokenizer  = tokenizer
        self.skill_vocab = skill_vocab
        self.max_len    = max_len

        print(f"[INFO] Encoding {len(texts)} resumes...")
        for text in texts:
            text = str(text)[:2000]      # cap at 2000 chars to stay within MAX_LEN
            words, word_labels = tag_resume(text, skill_vocab)
            if words:
                self.examples.append(self._encode(words, word_labels))

    def _encode(self, tokens: list, labels: list) -> dict:
        enc      = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        word_ids = enc.word_ids()

        aligned_labels = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)          # special tokens — ignored in loss
            elif wid != prev_word:
                lbl = labels[wid] if wid < len(labels) else "O"
                aligned_labels.append(LABEL2ID[lbl])
            else:
                lbl = labels[wid] if wid < len(labels) else "O"
                if lbl == "B-SKILL":
                    lbl = "I-SKILL"                  # subword continuation
                aligned_labels.append(LABEL2ID[lbl])
            prev_word = wid

        enc["labels"] = aligned_labels
        return {k: torch.tensor(v) for k, v in enc.items()}

    def __len__(self):        return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


# ── Data loader ───────────────────────────────────────────────────────────────

def load_resume_texts(resume_file: str = RESUME_FILE) -> list:
    """Loads resume texts from Person B's Resume.csv."""
    df = pd.read_csv(resume_file)

    # Find the resume text column
    text_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["resume_str", "resume", "text", "content"]):
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[-1]
        print(f"[WARN] Resume text column not found by name — using '{text_col}'")

    texts = df[text_col].dropna().tolist()
    print(f"[INFO] Loaded {len(texts)} resumes from {resume_file}")
    return texts


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(p):
    preds, label_ids = p
    preds = np.argmax(preds, axis=2)

    true_labels, pred_labels = [], []
    for pred_row, label_row in zip(preds, label_ids):
        tl, pl = [], []
        for p_val, l_val in zip(pred_row, label_row):
            if l_val != -100:
                tl.append(ID2LABEL[l_val])
                pl.append(ID2LABEL[p_val])
        true_labels.append(tl)
        pred_labels.append(pl)

    return {
        "f1":     f1_score(true_labels, pred_labels),
        "report": classification_report(true_labels, pred_labels),
    }


# ── Main training entry point ─────────────────────────────────────────────────

def main():
    # GPU check
    if torch.cuda.is_available():
        print(f"[INFO] GPU ready: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] No GPU found — training will be slow on CPU")

    # Load data
    texts      = load_resume_texts(RESUME_FILE)
    skill_vocab = load_skill_vocab(SKILLS_FILE)

    # Train / val split
    train_texts, val_texts = train_test_split(texts, test_size=VAL_SPLIT, random_state=42)
    print(f"[INFO] Train: {len(train_texts)} | Val: {len(val_texts)}")

    # Load tokenizer + model
    print(f"[INFO] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,    # replaces CoNLL head with our 3-label head
    )

    # Datasets
    train_ds = ResumeNERDataset(train_texts, tokenizer, skill_vocab)
    val_ds   = ResumeNERDataset(val_texts,   tokenizer, skill_vocab)

    # Training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n[INFO] Starting training — checkpoints saved after every epoch.")
    print("[INFO] Safe to leave overnight.\n")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[INFO] Model saved to {OUTPUT_DIR}")
    print("[INFO] Training complete!")


# ── Inference ─────────────────────────────────────────────────────────────────

_SKILL_KEYWORDS = [
    # Languages
    "Python","SQL","JavaScript","TypeScript","Java","C++","C#","C","Go","Rust","Scala",
    "PHP","Ruby","Swift","Kotlin","R","MATLAB","Bash","Shell","Perl","Groovy",

    # Web & Frontend
    "HTML","CSS","HTML5","CSS3","React","Vue","Angular","Next.js","Nuxt.js",
    "Node.js","Express","jQuery","Bootstrap","Tailwind","Sass","SCSS","Redux",
    "GraphQL","REST","REST APIs","gRPC","WebSockets","AJAX",

    # Backend & Auth
    "FastAPI","Flask","Django","Spring Boot","Laravel","Rails","ASP.NET",
    "JWT","OAuth","OAuth2","OpenID","SAML","Auth0","Passport.js","Keycloak",
    "Microservices","Serverless","API Gateway",

    # ML / AI
    "PyTorch","TensorFlow","Keras","Scikit-learn","XGBoost","LightGBM","CatBoost",
    "Hugging Face","BERT","GPT","LLM","NLP","Computer Vision","OpenCV",
    "Pandas","NumPy","Matplotlib","Seaborn","Plotly","SciPy","StatsModels",

    # Data Engineering
    "Kafka","Spark","PySpark","Dask","Flink","Airflow","Luigi","Prefect",
    "dbt","ETL","ELT","Data Pipeline","Hadoop","Hive","Presto","Trino",

    # Databases
    "PostgreSQL","MySQL","SQLite","Oracle","SQL Server","MongoDB","Redis",
    "Cassandra","DynamoDB","Elasticsearch","Neo4j","InfluxDB","Firestore",
    "Snowflake","BigQuery","Redshift","Databricks","Delta Lake",

    # Cloud & DevOps
    "AWS","GCP","Azure","EC2","S3","Lambda","RDS","ECS","EKS","CloudFormation",
    "Kubernetes","Docker","Terraform","Ansible","Helm","Jenkins","GitHub Actions",
    "CircleCI","ArgoCD","CI/CD","DevOps","SRE","Linux","Bash","Nginx","Apache",

    # MLOps
    "MLflow","Kubeflow","BentoML","Seldon","Feast","Tecton","DVC","Weights & Biases",
    "Pinecone","Weaviate","Chroma","FAISS","LangChain","LlamaIndex",

    # Tools & Practices
    "Git","GitHub","GitLab","Bitbucket","Jira","Confluence","Notion",
    "Tableau","Power BI","Looker","Metabase","Grafana","Prometheus","Datadog",
    "Postman","Swagger","OpenAPI","Jupyter","VS Code","PyCharm",

    # Testing & Security
    "Pytest","Jest","Selenium","Cypress","Playwright","JUnit","Mocha",
    "JWT","HTTPS","SSL","TLS","OWASP","Penetration Testing",

    # Concepts
    "Machine Learning","Deep Learning","Reinforcement Learning","Transfer Learning",
    "Object Detection","Text Classification","Sentiment Analysis","Recommendation System",
    "Feature Engineering","Model Deployment","A/B Testing","Data Warehousing",
    "Agile","Scrum","System Design","Microservices","Event-Driven Architecture",
]

def _keyword_skill_pass(text: str, already_seen: set) -> list:
    found = []
    text_lower = text.lower()
    for kw in _SKILL_KEYWORDS:
        if kw.lower() in text_lower and kw not in already_seen:
            found.append(kw)
            already_seen.add(kw)
    return found

def extract_entities(text: str, model_dir: str = OUTPUT_DIR) -> dict:
    """
    Runs NER inference on a raw resume string.
    Returns {"SKILL": [...], "EXPERIENCE": [...], "EDUCATION": [...]}

    Phase 1: if no trained model exists at model_dir, falls back to BASE_MODEL
             and applies CoNLL→resume label mapping + keyword supplement.
    Phase 2+: uses the fine-tuned model directly.
    """
    from transformers import pipeline

    using_base = not Path(model_dir).exists()
    active_model = BASE_MODEL if using_base else model_dir

    if using_base:
        print(f"[INFO] No trained model found — using base model for Phase 1 extraction.")

    pipe    = pipeline("ner", model=active_model, aggregation_strategy="simple")
    results = pipe(text)

    entities: dict = {"SKILL": [], "EXPERIENCE": [], "EDUCATION": []}
    seen = set()

    for r in results:
        label = r["entity_group"]
        word  = r["word"].strip()
        if not word or word in seen:
            continue
        seen.add(word)

        if label in entities:
            entities[label].append(word)
        elif label in ("MISC", "ORG"):
            entities["SKILL"].append(word)       # CoNLL phase-1 remap
        # PER and LOC ignored

    # Supplement with keyword pass (catches tools BASE_MODEL misses)
    entities["SKILL"].extend(_keyword_skill_pass(text, seen))

    return entities


if __name__ == "__main__":
    main()
