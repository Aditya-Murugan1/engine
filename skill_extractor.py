# skill_extractor.py
# ─────────────────────────────────────────────────────────────────────────────
# Fine-tunes dslim/bert-base-NER on Person B's real data.
#
# Data sources (from Person B):
#   data/Resume.csv   — 2484 resumes, column: "Resume_str" or last text column
#   data/Skills.xlsx  — 62580 O*NET skills, column: "Element Name"
#   data/train.jsonl  — hand-labelled sentences (seed data, all 7 label types)
#   data/val.jsonl    — hand-labelled validation sentences
#
# Label set (7 types — all active):
#   O, B-SKILL, I-SKILL, B-EXPERIENCE, I-EXPERIENCE, B-EDUCATION, I-EDUCATION
#
# Changes from previous version:
#   [1] LABELS extended from 3 to 7 — matches train/val.jsonl label set
#   [2] tag_resume() extended to tag EXPERIENCE + EDUCATION spans
#   [3] _keyword_skill_pass():
#         - \b → lookaround so C++, C#, CI/CD match correctly
#         - longest-first sort + substring guard so REST doesn't duplicate REST APIs
#         - C and R single-char keywords removed (too ambiguous, covered by NER)
#   [4] extract_entities() uses a module-level pipeline cache — model loads once
#   [5] ResumeNERDataset can now also ingest train.jsonl / val.jsonl as seed data
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
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
TRAIN_JSONL  = "data/train.jsonl"
VAL_JSONL    = "data/val.jsonl"
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
VAL_SPLIT    = 0.1

# ── Label schema (7 labels — skills + experience + education) ─────────────────
# FIX [1]: was ["O", "B-SKILL", "I-SKILL"] — mismatched with train/val.jsonl
# which contains B-EXPERIENCE, I-EXPERIENCE, B-EDUCATION, I-EDUCATION as well.

LABELS   = ["O", "B-SKILL", "I-SKILL", "B-EXPERIENCE", "I-EXPERIENCE", "B-EDUCATION", "I-EDUCATION"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ── Skill vocabulary loader ───────────────────────────────────────────────────

def load_skill_vocab(skills_file: str = SKILLS_FILE) -> set:
    """
    Loads the O*NET skills vocabulary from Person B's Skills.xlsx.
    Returns a set of lowercase skill strings (single words + bigrams).
    """
    df = pd.read_excel(skills_file)

    skill_col = None
    for col in df.columns:
        if col.strip().lower() == "element name":
            skill_col = col
            break
    if skill_col is None:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        skill_col = max(str_cols, key=lambda c: df[c].nunique())
        print(f"[WARN] 'Element Name' not found — using '{skill_col}' ({df[skill_col].nunique()} unique values)")

    vocab = set(df[skill_col].dropna().str.lower().str.strip().tolist())
    print(f"[INFO] Skill vocabulary: {len(vocab)} terms from {skills_file}")
    return vocab


# ── Pattern helpers for tagger ────────────────────────────────────────────────

# Job title words — used to detect EXPERIENCE spans (e.g. "Software Engineer at Infosys")
_TITLE_WORDS = {
    "engineer", "developer", "scientist", "analyst", "architect", "manager",
    "director", "lead", "head", "consultant", "specialist", "associate",
    "intern", "executive", "officer", "president", "vp", "cto", "ceo",
    "designer", "researcher", "administrator", "coordinator",
}

# Degree words — used to detect EDUCATION spans
_DEGREE_WORDS = {
    "bachelor", "master", "phd", "doctorate", "b.tech", "m.tech", "b.e",
    "m.e", "b.sc", "m.sc", "mba", "bba", "b.com", "m.com", "be", "me",
    "b.s", "m.s", "bs", "ms", "diploma",
}

# Section header lines to strip before regex processing in JD parser
_SECTION_HEADERS = re.compile(
    r"^(required skills?|nice to have|education|experience|about the role|"
    r"responsibilities|qualifications|what you.ll do|who you are|"
    r"benefits?|perks?|about us|the role|job title|company|location)\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ── Vocabulary-based BIO tagger ───────────────────────────────────────────────

def tag_resume(text: str, skill_vocab: set) -> tuple[list, list]:
    """
    Tags each word in a resume as B-SKILL/I-SKILL, B-EXPERIENCE/I-EXPERIENCE,
    B-EDUCATION/I-EDUCATION, or O using vocabulary + heuristic rules.

    FIX [2]: Previously only tagged SKILL. Now also tags EXPERIENCE and EDUCATION
    spans so ResumeNERDataset produces training signal for all 7 label types.

    Returns (words, labels).
    """
    words  = text.split()
    labels = ["O"] * len(words)
    i = 0

    while i < len(words):
        clean = words[i].lower().strip(".,;:()[]\"'–—-")

        # ── Bigram skill check (e.g. "machine learning", "deep learning") ──
        if i + 1 < len(words):
            clean_next = words[i + 1].lower().strip(".,;:()[]\"'–—-")
            bigram = clean + " " + clean_next
            if bigram in skill_vocab:
                labels[i]     = "B-SKILL"
                labels[i + 1] = "I-SKILL"
                i += 2
                continue

        # ── Single-word skill check ──
        if clean in skill_vocab:
            labels[i] = "B-SKILL"
            i += 1
            continue

        # ── EXPERIENCE span: title word + optional continuation ──
        # Pattern: "Software Engineer", "Senior Data Scientist at Infosys"
        if clean in _TITLE_WORDS:
            labels[i] = "B-EXPERIENCE"
            j = i + 1
            while j < len(words) and j < i + 5:
                w = words[j].lower().strip(".,;:()[]\"'–—-")
                if w in ("at", "—", "-", "infosys", "wipro", "google") or w.isdigit():
                    labels[j] = "I-EXPERIENCE"
                    j += 1
                else:
                    break
            i = j
            continue

        # ── EDUCATION span: degree word + continuation ──
        # Pattern: "Bachelor of Engineering", "MBA Marketing"
        if clean in _DEGREE_WORDS:
            labels[i] = "B-EDUCATION"
            j = i + 1
            while j < len(words) and j < i + 6:
                w = words[j].lower().strip(".,;:()[]\"'–—-")
                if w in ("of", "in", "and", "science", "engineering", "technology",
                         "arts", "commerce", "computer", "information", "marketing",
                         "finance", "economics", "mathematics", "statistics"):
                    labels[j] = "I-EDUCATION"
                    j += 1
                else:
                    break
            i = j
            continue

        i += 1

    return words, labels


# ── JSONL dataset (hand-labelled seed data) ───────────────────────────────────

class JsonlNERDataset(Dataset):
    """
    Loads pre-labelled sentences from train.jsonl / val.jsonl.
    Each line: {"tokens": [...], "labels": [...]}
    All 7 label types are valid here.

    FIX [5]: Previously this data was completely unused. Now merged with
    ResumeNERDataset via ConcatDataset so hand-labelled examples train the model.
    """
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = MAX_LEN):
        self.examples  = []
        self.tokenizer = tokenizer
        self.max_len   = max_len

        path = Path(jsonl_path)
        if not path.exists():
            print(f"[WARN] JSONL file not found: {jsonl_path} — skipping")
            return

        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"[INFO] Loading {len(lines)} labelled sentences from {jsonl_path}")

        for row in lines:
            tokens     = row["tokens"]
            str_labels = row["labels"]
            # Validate — skip rows with unknown labels
            if any(l not in LABEL2ID for l in str_labels):
                unknown = [l for l in str_labels if l not in LABEL2ID]
                print(f"[WARN] Skipping row with unknown labels: {unknown}")
                continue
            self.examples.append(self._encode(tokens, str_labels))

    def _encode(self, tokens: list, str_labels: list) -> dict:
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
                aligned_labels.append(-100)
            elif wid != prev_word:
                aligned_labels.append(LABEL2ID[str_labels[wid]])
            else:
                lbl = str_labels[wid]
                # Subword continuation: B- → I-
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                aligned_labels.append(LABEL2ID[lbl])
            prev_word = wid

        enc["labels"] = aligned_labels
        return {k: torch.tensor(v) for k, v in enc.items()}

    def __len__(self):        return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


# ── Resume auto-tag dataset ───────────────────────────────────────────────────

class ResumeNERDataset(Dataset):
    """
    Auto-tags raw resume strings using tag_resume().
    Used for the bulk of training data from Resume.csv.
    Now produces all 7 label types (SKILL, EXPERIENCE, EDUCATION).
    """
    def __init__(self, texts: list, tokenizer, skill_vocab: set, max_len: int = MAX_LEN):
        self.examples = []
        print(f"[INFO] Encoding {len(texts)} resumes...")
        for text in texts:
            text = str(text)[:2000]
            words, word_labels = tag_resume(text, skill_vocab)
            if words:
                self.examples.append(self._encode(words, word_labels, tokenizer, max_len))

    def _encode(self, tokens: list, labels: list, tokenizer, max_len: int) -> dict:
        enc      = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        word_ids = enc.word_ids()

        aligned_labels = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word:
                lbl = labels[wid] if wid < len(labels) else "O"
                aligned_labels.append(LABEL2ID[lbl])
            else:
                lbl = labels[wid] if wid < len(labels) else "O"
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                aligned_labels.append(LABEL2ID[lbl])
            prev_word = wid

        enc["labels"] = aligned_labels
        return {k: torch.tensor(v) for k, v in enc.items()}

    def __len__(self):        return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_resume_texts(resume_file: str = RESUME_FILE) -> list:
    df = pd.read_csv(resume_file)
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
    if torch.cuda.is_available():
        print(f"[INFO] GPU ready: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] No GPU found — training will be slow on CPU")

    texts       = load_resume_texts(RESUME_FILE)
    skill_vocab = load_skill_vocab(SKILLS_FILE)

    train_texts, val_texts = train_test_split(texts, test_size=VAL_SPLIT, random_state=42)
    print(f"[INFO] Train: {len(train_texts)} | Val: {len(val_texts)}")

    print(f"[INFO] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Auto-tag dataset from Resume.csv
    resume_train_ds = ResumeNERDataset(train_texts, tokenizer, skill_vocab)
    resume_val_ds   = ResumeNERDataset(val_texts,   tokenizer, skill_vocab)

    # FIX [5]: Merge hand-labelled JSONL seed data with auto-tagged dataset
    jsonl_train_ds = JsonlNERDataset(TRAIN_JSONL, tokenizer)
    jsonl_val_ds   = JsonlNERDataset(VAL_JSONL,   tokenizer)

    train_ds = ConcatDataset([resume_train_ds, jsonl_train_ds]) if len(jsonl_train_ds) else resume_train_ds
    val_ds   = ConcatDataset([resume_val_ds,   jsonl_val_ds])   if len(jsonl_val_ds)   else resume_val_ds

    print(f"[INFO] Final train size: {len(train_ds)} | val size: {len(val_ds)}")

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


# ── Inference keyword list ────────────────────────────────────────────────────
# Note: single-char keywords "C" and "R" removed — too ambiguous, NER handles them.
# Note: ordered longest-first at runtime in _keyword_skill_pass to prevent
#       shorter substrings ("REST") duplicating longer matches ("REST APIs").

_SKILL_KEYWORDS = [
    # Languages (multi-char only)
    "Python", "SQL", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go",
    "Rust", "Scala", "PHP", "Ruby", "Swift", "Kotlin", "MATLAB", "Bash",
    "Shell", "Perl", "Groovy",

    # Web & Frontend
    "HTML", "CSS", "HTML5", "CSS3", "React", "Vue", "Angular", "Next.js",
    "Nuxt.js", "Node.js", "Express", "jQuery", "Bootstrap", "Tailwind",
    "Sass", "SCSS", "Redux", "GraphQL", "REST APIs", "gRPC", "WebSockets", "AJAX",

    # Backend & Auth
    "FastAPI", "Flask", "Django", "Spring Boot", "Laravel", "Rails", "ASP.NET",
    "JWT", "OAuth", "OAuth2", "OpenID", "SAML", "Auth0", "Passport.js",
    "Keycloak", "Microservices", "Serverless", "API Gateway",

    # ML / AI
    "PyTorch", "TensorFlow", "Keras", "Scikit-learn", "XGBoost", "LightGBM",
    "CatBoost", "Hugging Face", "BERT", "GPT", "LLM", "NLP",
    "Computer Vision", "OpenCV", "Pandas", "NumPy", "Matplotlib", "Seaborn",
    "Plotly", "SciPy", "StatsModels",

    # Data Engineering
    "Kafka", "Spark", "PySpark", "Dask", "Flink", "Airflow", "Luigi",
    "Prefect", "dbt", "ETL", "ELT", "Data Pipeline", "Hadoop", "Hive",
    "Presto", "Trino",

    # Databases
    "PostgreSQL", "MySQL", "SQLite", "Oracle", "SQL Server", "MongoDB",
    "Redis", "Cassandra", "DynamoDB", "Elasticsearch", "Neo4j", "InfluxDB",
    "Firestore", "Snowflake", "BigQuery", "Redshift", "Databricks", "Delta Lake",

    # Cloud & DevOps
    "AWS", "GCP", "Azure", "EC2", "S3", "Lambda", "RDS", "ECS", "EKS",
    "CloudFormation", "Kubernetes", "Docker", "Terraform", "Ansible", "Helm",
    "Jenkins", "GitHub Actions", "CircleCI", "ArgoCD", "CI/CD", "DevOps",
    "SRE", "Linux", "Nginx", "Apache",

    # MLOps
    "MLflow", "Kubeflow", "BentoML", "Seldon", "Feast", "Tecton", "DVC",
    "Weights & Biases", "Pinecone", "Weaviate", "Chroma", "FAISS",
    "LangChain", "LlamaIndex",

    # Tools & Practices
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence", "Notion",
    "Tableau", "Power BI", "Looker", "Metabase", "Grafana", "Prometheus",
    "Datadog", "Postman", "Swagger", "OpenAPI", "Jupyter", "VS Code",
    "PyCharm",

    # Testing & Security
    "Pytest", "Jest", "Selenium", "Cypress", "Playwright", "JUnit", "Mocha",
    "HTTPS", "SSL", "TLS", "OWASP", "Penetration Testing",

    # Concepts
    "Machine Learning", "Deep Learning", "Reinforcement Learning",
    "Transfer Learning", "Object Detection", "Text Classification",
    "Sentiment Analysis", "Recommendation System", "Feature Engineering",
    "Model Deployment", "A/B Testing", "Data Warehousing", "Agile", "Scrum",
    "System Design", "Event-Driven Architecture",
]


def _keyword_skill_pass(text: str, already_seen: set) -> list:
    """
    Scans text for known skill keywords and returns those not already seen.

    FIX [3a]: Uses lookaround (?<![a-zA-Z0-9]) / (?![a-zA-Z0-9]) instead of
              \\b so that C++, C#, CI/CD, gRPC all match correctly.
              \\b fails for keywords ending/starting with non-word chars.

    FIX [3b]: Iterates keywords longest-first and skips any keyword whose
              lowercase form is a substring of an already-found keyword.
              Prevents "REST" from duplicating "REST APIs".

    FIX [3c]: Single-char keywords "C" and "R" removed from _SKILL_KEYWORDS.
              They caused false positives (e.g. "R" in "Research") and the
              NER pipeline handles them in context.
    """
    found    = []
    tl       = " " + text.lower() + " "   # padding for start/end boundary

    for kw in sorted(_SKILL_KEYWORDS, key=len, reverse=True):   # FIX [3b]: longest first
        if kw in already_seen:
            continue
        kw_lower = kw.lower()
        # FIX [3b]: skip if this kw is contained in a longer match already found
        if any(kw_lower in f.lower() for f in found):
            continue
        # FIX [3a]: lookaround instead of \b
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(kw_lower) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, tl):
            found.append(kw)
            already_seen.add(kw)

    return found


# ── Inference pipeline cache ──────────────────────────────────────────────────
# FIX [4]: pipeline was re-loaded from disk on every call to extract_entities().
# _NER_PIPELINE caches by model path so the model loads exactly once per process.

_NER_PIPELINE: dict = {}


def extract_entities(text: str, model_dir: str = OUTPUT_DIR) -> dict:
    """
    Runs NER inference on a raw resume string.
    Returns {"SKILL": [...], "EXPERIENCE": [...], "EDUCATION": [...]}

    Phase 1: if no trained model exists at model_dir, falls back to BASE_MODEL
             with CoNLL → resume label mapping + keyword supplement.
    Phase 2+: uses the fine-tuned 7-label model directly.
    """
    from transformers import pipeline as hf_pipeline

    using_base   = not Path(model_dir).exists()
    active_model = BASE_MODEL if using_base else model_dir

    if using_base:
        print("[INFO] No trained model found — using base model for Phase 1 extraction.")

    # FIX [4]: load once, reuse every call
    if active_model not in _NER_PIPELINE:
        print(f"[INFO] Loading NER pipeline from {active_model} (one-time)...")
        _NER_PIPELINE[active_model] = hf_pipeline(
            "ner", model=active_model, aggregation_strategy="simple"
        )

    pipe    = _NER_PIPELINE[active_model]
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
            # CoNLL phase-1 fallback: remap org/misc to SKILL
            entities["SKILL"].append(word)
        # PER and LOC intentionally ignored

    # Supplement NER output with deterministic keyword scan
    entities["SKILL"].extend(_keyword_skill_pass(text, seen))

    return entities


if __name__ == "__main__":
    main()
