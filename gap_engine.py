# gap_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Compares extracted resume entities against a JD (primary) or O*NET (fallback).
# Outputs: skill gaps + course recommendations + training_hints for launcher.
#
# ── INPUT CONTRACTS ──────────────────────────────────────────────────────────
# extracted_entities : output of skill_extractor.extract_entities()
#   {"SKILL": [...], "EXPERIENCE": [...], "EDUCATION": [...]}
#
# jd_text           : raw job description string (or None → triggers O*NET fallback)
#
# onet_csv          : path to O*NET skills CSV (data/onet_skills.csv)
#   Required columns: "Element Name", "Title"
#
# courses_json      : path to Person B's course catalog (data/courses.json)
#   [{"id": "c1", "title": "...", "skills_covered": ["Python", "SQL", ...]}]
#
# ── OUTPUT: gap_report.json ───────────────────────────────────────────────────
# {
#   "gaps": [{"skill": "Kubernetes", "score": 0.21, "source": "JD"}],
#   "recommendations": [{"gap_skill": "...", "course_id": "...",
#                         "course_title": "...", "match_score": 0.87}],
#   "training_hints": {
#     "class_weights": {"O": 1.0, "SKILL": 2.4, "EXPERIENCE": 1.1, "EDUCATION": 1.0}
#   }
# }
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL      = "all-MiniLM-L6-v2"
GAP_THRESHOLD    = 0.50
COURSE_THRESHOLD = 0.35   # lowered further — MiniLM similarity for short skill→course pairs
TOP_K_GAPS       = 15
ONET_ROLE_COL    = "Title"
ONET_SKILL_COL   = "Element Name"

# ── Embedding cache ───────────────────────────────────────────────────────────

class EmbedCache:
    _model  = None
    _cache: dict = {}

    @classmethod
    def model(cls) -> SentenceTransformer:
        if cls._model is None:
            print(f"  Loading embedding model ({EMBED_MODEL})...")
            cls._model = SentenceTransformer(EMBED_MODEL)
        return cls._model

    @classmethod
    def embed(cls, texts: list) -> np.ndarray:
        missing = [t for t in texts if t not in cls._cache]
        if missing:
            vecs = cls.model().encode(missing, show_progress_bar=False,
                                      normalize_embeddings=True)
            for t, v in zip(missing, vecs):
                cls._cache[t] = v
        return np.stack([cls._cache[t] for t in texts])

# ── JD skill extraction ───────────────────────────────────────────────────────

_SKILL_PATTERNS = [
    r"\b[A-Z][a-zA-Z+#\.]+(?:\s[A-Z][a-zA-Z+#\.]+){0,2}\b",
    r"\b(?:proficiency|experience|knowledge|familiarity)\s+(?:in|with)\s+([\w\s/+#]+)",
]

# Noise words that leak through the regex — prose, section headers, company names
_JD_STOPWORDS = {
    # Generic prose / section headers
    "have", "nice", "mid", "job", "title", "level", "about", "role", "note",
    "required", "skills", "experience", "education", "company", "team", "you",
    "will", "our", "and", "the", "for", "with", "from", "this", "that", "are",
    "also", "join", "build", "work", "using", "looking", "strong", "good",
    "understanding", "knowledge", "familiarity", "proficiency", "ability",
    "we", "who", "including", "such", "value", "sector", "monitoring",
    "reporting", "planning", "support", "development", "strategy", "process",
    "service", "quality", "performance", "operations", "communication",
    "research", "design", "testing", "review", "training", "management",
    # Qualifications that are not skills
    "mba", "cfa", "ca", "phd", "bachelor", "master", "degree", "preferred",
    # Job titles bleeding through
    "analyst", "manager", "engineer", "senior", "junior", "associate",
    "director", "head", "lead", "intern",
    # Country/generic adjectives
    "indian", "strong", "large", "deep", "broad",
    # Company names
    "kedaara", "capital", "meesho", "zepto", "nykaa", "techcorp",
    "icici", "hdfc", "flipkart", "swiggy", "india", "tech", "corp",
}

# Multi-word phrases that are noise even if each word looks like a skill
_JD_NOISE_PHRASES = {
    "kedaara capital", "private equity analyst", "cfa level",
    "nice to have", "about the role", "required skills",
    "strong sql", "strong python", "deep expertise",
    "senior ml engineer", "junior ml engineer", "ml engineer",
    "growth marketing manager", "marketing manager", "product manager",
    "investment banking analyst", "data scientist", "data engineer",
    "software engineer", "frontend engineer", "backend engineer",
    "hubspot crm", "excel advanced", "advanced excel",
    "google analytics 4",  # keep as skill but not as phrase-noise
}

# Job title patterns — multi-word phrases ending in a title word
_TITLE_WORDS = {
    "engineer", "manager", "analyst", "developer", "designer",
    "scientist", "architect", "consultant", "specialist", "lead",
    "director", "head", "officer", "associate", "intern", "executive",
}

# Known tech skill keywords to scan directly — ensures nothing important is missed
_JD_SKILL_KEYWORDS = [
    # Programming
    "Python","SQL","Java","JavaScript","TypeScript","C++","C#","Scala","Rust",
    "MATLAB","VBA","Bash","PHP","Ruby","Swift","Kotlin",
    # ML / AI
    "PyTorch","TensorFlow","Keras","Scikit-learn","Hugging Face","BERT","NLP",
    "XGBoost","LightGBM","CatBoost","LangChain","LlamaIndex","OpenCV",
    # Data Engineering
    "MLflow","Airflow","Feast","Pinecone","Weaviate","Kafka","Spark","Dask",
    "dbt","Hadoop","Hive","Flink","Prefect","Delta Lake",
    # Cloud & DevOps
    "Kubernetes","Docker","Terraform","AWS","GCP","Azure","CI/CD","DevOps",
    "EC2","S3","Lambda","SageMaker","GitHub Actions","Jenkins","Ansible","Helm",
    # Databases
    "Snowflake","BigQuery","PostgreSQL","MySQL","Redis","MongoDB","Elasticsearch",
    "Cassandra","DynamoDB","Redshift","Databricks","Firestore","Neo4j",
    # Web
    "React","Node.js","FastAPI","Flask","Django","GraphQL","REST APIs","JWT",
    "OAuth","Next.js","Vue","Angular","Spring Boot",
    # Tools
    "Git","Linux","Jupyter","Postman","Swagger","Tableau","Power BI","Grafana",
    "Datadog","Prometheus","Looker","Metabase","feature store","transformers",
    # Finance domain
    "DCF","LBO","Bloomberg","Bloomberg Terminal","Capital IQ","Excel","VBA",
    "IFRS","Ind AS","Financial Modelling","Valuation","Due Diligence",
    "Credit Analysis","Debt Structuring","Portfolio Management","ESG",
    "Comparable Company Analysis","Precedent Transactions","Pitch Books",
    "Financial Statement Analysis","M&A","Private Equity","Hedge Fund",
    "Risk Management","Derivatives","Fixed Income","Equity Research",
    "Quantitative Finance","Algorithmic Trading","FRM","CFA","PowerPoint",
    # Marketing domain
    "SEO","SEM","Google Ads","Meta Ads","Facebook Ads","Email Marketing",
    "HubSpot","Salesforce","Marketo","Braze","Clevertap","Mailchimp",
    "Google Analytics","Google Analytics 4","Mixpanel","Amplitude","Heap",
    "A/B Testing","Optimizely","SEMrush","Ahrefs","Moz","Content Strategy",
    "Social Media Marketing","Influencer Marketing","Performance Marketing",
    "Growth Marketing","CRM","Attribution Modelling","AppsFlyer","Adjust",
    "App Store Optimisation","ASO","Programmatic Advertising","Looker",
    "Cohort Analysis","Retention Marketing","Push Notifications",
    "Lifecycle Marketing","CDPs","Segment","mParticle",
    # Design / Product
    "Figma","Sketch","Adobe XD","Canva","InVision","Zeplin",
    "Product Management","User Research","Wireframing","Prototyping",
    "JIRA","Confluence","Notion","Miro","Agile","Scrum","Kanban",
    # Healthcare / Science
    "SPSS","SAS","Clinical Trials","HIPAA","HL7","FHIR","Tableau",
    # Legal / Compliance
    "GDPR","ISO 27001","SOC 2","PCI DSS","Compliance","Due Diligence",
]

def extract_skills_from_jd(jd_text: str) -> list:
    """
    Extracts skill-like terms from a JD using regex + keyword scan.
    Filters out prose noise (section headers, common verbs, company names).
    """
    # Collapse newlines/tabs to spaces so multi-line JDs don't create split tokens
    jd_text = " ".join(jd_text.split())
    candidates = set()

    # Regex pass — catches CamelCase/acronyms and "experience with X" patterns
    for pattern in _SKILL_PATTERNS:
        for match in re.finditer(pattern, jd_text):
            term = (match.group(1) if match.lastindex else match.group()).strip().replace("\n", " ").replace("\r", "")
            term_words = term.lower().split()
            if (2 < len(term) < 40
                    and term.lower() not in _JD_STOPWORDS
                    and term.lower() not in _JD_NOISE_PHRASES
                    and not all(w in _JD_STOPWORDS for w in term_words)
                    # Filter job title phrases: "Senior ML Engineer", "Growth Marketing Manager"
                    and not (len(term_words) >= 2 and term_words[-1] in _TITLE_WORDS)
                    and any(c.isupper() or c in "+#./0123456789-" for c in term)):
                candidates.add(term)

    # Keyword scan pass — catches known tools the regex might miss
    for kw in _JD_SKILL_KEYWORDS:
        if kw.lower() in jd_text.lower():
            candidates.add(kw)

    return list(candidates)

# ── O*NET fallback ────────────────────────────────────────────────────────────

def load_onet_skills(onet_csv: str, role_query: str = None) -> list:
    # Accept both .xlsx (Person B's Skills.xlsx) and .csv (our sample onet_skills.csv)
    if onet_csv.endswith(".xlsx"):
        df = pd.read_excel(onet_csv)
        # Normalise column names to match expected ONET_SKILL_COL / ONET_ROLE_COL
        col_map = {}
        for col in df.columns:
            if "element" in col.lower() and "name" in col.lower():
                col_map[col] = ONET_SKILL_COL
            elif "title" in col.lower():
                col_map[col] = ONET_ROLE_COL
        df = df.rename(columns=col_map)
    else:
        df = pd.read_csv(onet_csv)
    if role_query and ONET_ROLE_COL in df.columns:
        roles     = df[ONET_ROLE_COL].dropna().unique().tolist()
        role_vecs = EmbedCache.embed(roles)
        query_vec = EmbedCache.embed([role_query])
        sims      = cosine_similarity(query_vec, role_vecs)[0]
        best_role = roles[int(np.argmax(sims))]
        print(f"  O*NET best-match role: '{best_role}'")
        df = df[df[ONET_ROLE_COL] == best_role]
    return df[ONET_SKILL_COL].dropna().unique().tolist()

# ── Core gap computation ──────────────────────────────────────────────────────

def compute_gaps(resume_skills: list, required_skills: list,
                 threshold: float = GAP_THRESHOLD) -> list:
    if not resume_skills or not required_skills:
        return [{"skill": s, "score": 0.0} for s in required_skills]

    res_vecs   = EmbedCache.embed(resume_skills)
    req_vecs   = EmbedCache.embed(required_skills)
    sim_matrix = cosine_similarity(req_vecs, res_vecs)
    max_sims   = sim_matrix.max(axis=1)

    gaps = []
    for skill, score in zip(required_skills, max_sims):
        if score < threshold:
            gaps.append({"skill": skill, "score": round(float(score), 4)})

    return sorted(gaps, key=lambda x: x["score"])[:TOP_K_GAPS]

# ── Course matching ───────────────────────────────────────────────────────────

def match_courses(gaps: list, courses_json: str,
                  threshold: float = COURSE_THRESHOLD) -> list:
    courses = json.loads(Path(courses_json).read_text())
    if not courses or not gaps:
        return []

    # Build lowercase lookup: exact skill name → course index
    course_skills_lower = [
        {s.lower() for s in c.get("skills_covered", [])} for c in courses
    ]

    # Build embedding centroids for semantic fallback
    course_vecs = []
    for c in courses:
        skill_texts = c.get("skills_covered", [c["title"]])
        vecs        = EmbedCache.embed(skill_texts)
        course_vecs.append(vecs.mean(axis=0))
    course_matrix = np.stack(course_vecs)

    gap_skills = [g["skill"] for g in gaps]
    gap_vecs   = EmbedCache.embed(gap_skills)
    sims       = cosine_similarity(gap_vecs, course_matrix)

    recommendations = []
    used_courses = set()   # each course appears only once across all gaps

    for i, gap in enumerate(gaps):
        gap_lower    = gap["skill"].lower()
        gap_variants = {gap_lower, gap_lower + "s", gap_lower.rstrip("s")}

        # Strategy 1: exact match — skip if course already used
        exact_idx = next(
            (j for j, skill_set in enumerate(course_skills_lower)
             if gap_variants & skill_set and j not in used_courses),
            None
        )
        if exact_idx is not None:
            recommendations.append({
                "gap_skill":    gap["skill"],
                "course_id":    courses[exact_idx]["id"],
                "course_title": courses[exact_idx]["title"],
                "match_score":  1.0,
            })
            used_courses.add(exact_idx)
            continue

        # Strategy 2: semantic similarity — pick best unused course
        ranked = np.argsort(sims[i])[::-1]
        for best_idx in ranked:
            idx = int(best_idx)
            if idx in used_courses:
                continue
            best_score = float(sims[i][idx])
            if best_score >= threshold:
                recommendations.append({
                    "gap_skill":    gap["skill"],
                    "course_id":    courses[idx]["id"],
                    "course_title": courses[idx]["title"],
                    "match_score":  round(best_score, 4),
                })
                used_courses.add(idx)
            break

    return sorted(recommendations, key=lambda x: x["match_score"], reverse=True)

# ── Training hints ────────────────────────────────────────────────────────────

def compute_training_hints(gaps: list, extracted_entities: dict) -> dict:
    n_gaps   = len(gaps)
    n_skills = len(extracted_entities.get("SKILL", []))
    n_exp    = len(extracted_entities.get("EXPERIENCE", []))
    n_edu    = len(extracted_entities.get("EDUCATION", []))

    raw = {
        "SKILL":      1.0 + (n_gaps / max(n_skills, 1)),
        "EXPERIENCE": 1.0 + (1.0 / max(n_exp, 1)),
        "EDUCATION":  1.0 + (1.0 / max(n_edu, 1)),
    }
    mean_w  = np.mean(list(raw.values()))
    weights = {k: round(v / mean_w, 4) for k, v in raw.items()}
    weights["O"] = 1.0
    return {"class_weights": weights}

# ── Main entry point ──────────────────────────────────────────────────────────

def run_gap_engine(
    extracted_entities: dict,
    onet_csv:           str,
    courses_json:       str,
    jd_text:            str  = None,
    role_query:         str  = None,
    output_path:        str  = "data/gap_report.json",
) -> dict:
    resume_skills = extracted_entities.get("SKILL", [])

    # Step 1: required skills
    if jd_text:
        required_skills = extract_skills_from_jd(jd_text)
        source_label    = "JD"
        print(f"  Extracted {len(required_skills)} skills from JD")
        if len(required_skills) < 10:
            print("  JD sparse — supplementing with O*NET")
            onet_skills     = load_onet_skills(onet_csv, role_query)
            required_skills = list(set(required_skills + onet_skills))
            source_label    = "JD+ONET"
    else:
        print("  No JD — using O*NET fallback")
        required_skills = load_onet_skills(onet_csv, role_query)
        source_label    = "ONET"

    # Step 2: gaps
    gaps = compute_gaps(resume_skills, required_skills)
    for g in gaps:
        g["source"] = source_label

    # Step 3: course recommendations
    recommendations = match_courses(gaps, courses_json)

    # Step 4: training hints
    training_hints = compute_training_hints(gaps, extracted_entities)

    # Step 5: write report
    report = {
        "gaps":            gaps,
        "recommendations": recommendations,
        "training_hints":  training_hints,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2))
    print(f"  Gap report written → {output_path}")
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from skill_extractor import extract_entities

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",  required=True)
    parser.add_argument("--jd",      default=None)
    parser.add_argument("--role",    default=None)
    parser.add_argument("--onet",    default="data/onet_skills.csv")
    parser.add_argument("--courses", default="data/courses.json")
    parser.add_argument("--output",  default="data/gap_report.json")
    args = parser.parse_args()

    resume_text = Path(args.resume).read_text()
    jd_text     = Path(args.jd).read_text() if args.jd else None
    entities    = extract_entities(resume_text)
    report      = run_gap_engine(
        extracted_entities=entities,
        jd_text=jd_text,
        role_query=args.role,
        onet_csv=args.onet,
        courses_json=args.courses,
        output_path=args.output,
    )
    print(f"\nTop gaps : {[g['skill'] for g in report['gaps'][:5]]}")
    print(f"Weights  : {report['training_hints']['class_weights']}")
