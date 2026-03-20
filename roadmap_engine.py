# roadmap_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts a gap_report + matched_skills into a fully-ordered, dependency-aware
# learning roadmap for a specific candidate.
#
# Key design decisions vs the old build_roadmap() in api.py:
#   [1] Gap score drives priority — not a static SKILL_DEPS tier string.
#       score < 0.30  → high   (must learn before applying)
#       0.30–0.45     → medium (bridge while applying)
#       > 0.45        → low    (nice-to-have / learn on the job)
#
#   [2] Already-known skills are filtered OUT of the plan.
#       Prerequisites that exist in matched_skills are not added as nodes;
#       they are referenced as "unlocks_from" so the UI can show context.
#
#   [3] Course deduplication is enforced: one course per roadmap week-block,
#       one week-block per gap skill. Multiple gaps mapped to the same course
#       are grouped into a single node (e.g. NLP + BERT + Hugging Face → c02).
#
#   [4] Topological sort respects dependency edges so prerequisites always
#       appear before the skills that need them.
#
#   [5] Week estimates are computed from DURATIONS, summed per tier, and
#       returned as a timeline the UI can render.
#
#   [6] Noise skills ("Tecton Nice", "Weaviate Education", "Engineer Company")
#       are filtered before the plan is built using a blocklist + regex check.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import re
import json
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Priority thresholds ───────────────────────────────────────────────────────

THRESHOLD_HIGH   = 0.30   # score < this  → high priority
THRESHOLD_MEDIUM = 0.45   # score < this  → medium priority
                           # score >= this → low priority


# ── Skill dependency graph ────────────────────────────────────────────────────
# Format: skill → (tier_hint, [prerequisites])
# tier_hint is now only a tiebreak; gap score is the real driver.

SKILL_DEPS: dict[str, tuple[str, list[str]]] = {
    # Foundations
    "Python":              ("foundation", []),
    "SQL":                 ("foundation", []),
    "Git":                 ("foundation", []),
    "Linux":               ("foundation", []),
    "Docker":              ("foundation", []),
    "Excel":               ("foundation", []),
    "Statistics":          ("foundation", []),
    "Financial Modelling": ("foundation", []),

    # Core ML
    "PyTorch":             ("core", ["Python"]),
    "TensorFlow":          ("core", ["Python"]),
    "Keras":               ("core", ["Python", "TensorFlow"]),
    "Scikit-learn":        ("core", ["Python"]),
    "XGBoost":             ("core", ["Python", "Scikit-learn"]),
    "NLP":                 ("core", ["Python"]),
    "BERT":                ("core", ["NLP", "Hugging Face"]),
    "Hugging Face":        ("core", ["Python", "PyTorch"]),
    "transformers":        ("core", ["Python", "PyTorch"]),
    "Feature Engineering": ("core", ["Python", "Scikit-learn"]),
    "MLflow":              ("core", ["Python"]),
    "Model Deployment":    ("core", ["Docker", "PyTorch"]),

    # DevOps / Infra
    "Kubernetes":          ("core", ["Docker"]),
    "CI/CD":               ("core", ["Git", "Docker"]),
    "Terraform":           ("core", ["Docker"]),
    "GitHub Actions":      ("core", ["Git"]),
    "Jenkins":             ("core", ["Git"]),
    "AWS":                 ("core", []),
    "GCP":                 ("core", []),
    "Azure":               ("core", []),
    "SageMaker":           ("advanced", ["AWS", "PyTorch"]),

    # Data Engineering
    "Snowflake":           ("core", ["SQL"]),
    "BigQuery":            ("core", ["SQL"]),
    "Redshift":            ("core", ["SQL"]),
    "dbt":                 ("core", ["SQL"]),
    "Kafka":               ("advanced", ["Python"]),
    "Spark":               ("advanced", ["Python"]),
    "PySpark":             ("advanced", ["Python", "Spark"]),
    "Dask":                ("advanced", ["Python"]),
    "Airflow":             ("advanced", ["Python"]),
    "Flink":               ("advanced", ["Python"]),

    # MLOps / Vector DBs
    "Feast":               ("advanced", ["Python"]),
    "Tecton":              ("advanced", ["Python"]),
    "feature store":       ("advanced", ["Python"]),
    "Pinecone":            ("optional", ["Python"]),
    "Weaviate":            ("optional", ["Python"]),
    "Chroma":              ("optional", ["Python"]),
    "FAISS":               ("optional", ["Python"]),
    "LangChain":           ("optional", ["Python", "Hugging Face"]),
    "LlamaIndex":          ("optional", ["Python", "Hugging Face"]),
    "DVC":                 ("optional", ["Git"]),
    "Kubeflow":            ("optional", ["Kubernetes", "Python"]),

    # Finance
    "DCF":                 ("core", ["Excel", "Financial Modelling"]),
    "LBO":                 ("core", ["Excel", "Financial Modelling"]),
    "Credit Analysis":     ("core", ["Financial Modelling"]),
    "Bloomberg Terminal":  ("core", []),
    "Capital IQ":          ("core", []),
    "ESG":                 ("advanced", ["Financial Modelling"]),

    # Marketing
    "Google Ads":          ("core", []),
    "Meta Ads":            ("core", []),
    "Google Analytics":    ("core", []),
    "HubSpot":             ("core", []),
    "Salesforce":          ("core", []),
    "SEO":                 ("core", []),
    "Attribution Modelling": ("advanced", ["Google Analytics"]),
    "Cohort Analysis":     ("advanced", ["Google Analytics", "SQL"]),
    "AppsFlyer":           ("advanced", ["Google Analytics"]),
}


# ── Duration estimates (weeks) ────────────────────────────────────────────────

DURATIONS: dict[str, float] = {
    "PyTorch": 3, "TensorFlow": 3, "Keras": 2, "Scikit-learn": 2,
    "NLP": 3, "BERT": 2, "Hugging Face": 2, "transformers": 2,
    "Feature Engineering": 2, "MLflow": 1, "Model Deployment": 2,
    "Docker": 1, "Kubernetes": 2, "CI/CD": 1, "Terraform": 2,
    "GitHub Actions": 0.5, "AWS": 2, "SageMaker": 2,
    "Snowflake": 1, "BigQuery": 1, "dbt": 1,
    "Kafka": 2, "Spark": 2, "PySpark": 2, "Dask": 1.5, "Airflow": 2,
    "Feast": 1, "Tecton": 1, "feature store": 1,
    "Pinecone": 0.5, "Weaviate": 1, "FAISS": 0.5, "LangChain": 1,
    "DCF": 2, "LBO": 3, "Financial Modelling": 3, "Credit Analysis": 2,
    "Bloomberg Terminal": 1, "Capital IQ": 1, "ESG": 1,
    "Google Ads": 1, "Meta Ads": 1, "SEO": 2, "Google Analytics": 1,
    "HubSpot": 1, "Salesforce": 1, "Attribution Modelling": 2,
    "Cohort Analysis": 1, "AppsFlyer": 1,
}


# ── Skill descriptions ────────────────────────────────────────────────────────

DESCRIPTIONS: dict[str, str] = {
    "PyTorch":             "Tensors, autograd, and neural network training from scratch.",
    "TensorFlow":          "Build and train deep learning models with TF and Keras.",
    "Hugging Face":        "Pretrained transformers for NLP — fine-tuning and pipelines.",
    "BERT":                "Fine-tune BERT for classification, NER, and Q&A tasks.",
    "transformers":        "Transformer architecture, attention, and HF library usage.",
    "NLP":                 "Tokenisation, embeddings, sequence models, text pipelines.",
    "MLflow":              "Experiment tracking, model registry, run comparison.",
    "Docker":              "Containerise apps for reproducible, portable deployments.",
    "Kubernetes":          "Orchestrate containers at scale for production ML services.",
    "CI/CD":               "Automate build, test, and deploy pipelines for faster delivery.",
    "Airflow":             "Schedule and monitor ML and data pipelines as DAGs.",
    "Kafka":               "Real-time event-driven data pipelines and stream processing.",
    "Spark":               "Distributed data processing and large-scale analytics.",
    "Dask":                "Parallel computing in Python for out-of-memory datasets.",
    "Feast":               "Feature store — manage, serve, and share ML features.",
    "Tecton":              "Enterprise feature platform for real-time ML feature serving.",
    "feature store":       "Concepts and tools for managing ML features at scale.",
    "Snowflake":           "Cloud data warehousing and SQL analytics at scale.",
    "BigQuery":            "Large-scale analytics on GCP with BigQuery SQL.",
    "Weaviate":            "Vector database for semantic search and RAG pipelines.",
    "Pinecone":            "Managed vector database for production similarity search.",
    "SageMaker":           "End-to-end ML on AWS: training, tuning, and deployment.",
    "AWS":                 "Core cloud services: compute, storage, networking, ML.",
    "Feature Engineering": "Transform raw data into features that improve model performance.",
    "Model Deployment":    "Serve ML models in production via APIs and cloud platforms.",
    "DCF":                 "Discounted Cash Flow modelling for company and asset valuation.",
    "LBO":                 "Leveraged Buyout modelling for private equity transactions.",
    "Google Analytics":    "Funnel analysis, event tracking, and audience insights on GA4.",
    "SEO":                 "Technical SEO, keyword strategy, and on-page optimisation.",
    "HubSpot":             "CRM workflows, email sequences, and marketing automation.",
}


# ── Noise filter ──────────────────────────────────────────────────────────────
# Catches artefacts like "Tecton Nice", "Weaviate Education", "Engineer Company"

_NOISE_WORDS = {
    "nice", "education", "company", "corp", "techcorp", "india", "indian",
    "level", "required", "about", "role", "title", "job", "senior", "junior",
    "engineer", "manager", "analyst", "lead", "director", "associate",
}

_MULTI_WORD_NOISE = re.compile(
    r"^(nice to have|about the role|required skills?|"
    r"bachelor|master|phd|mba|degree|"
    r"engineer company|tech corp|techcorp)$",
    re.IGNORECASE,
)


def _is_noise(skill: str) -> bool:
    """Return True if skill is a JD parsing artefact, not a real skill."""
    sl = skill.lower().strip()
    if _MULTI_WORD_NOISE.match(sl):
        return True
    words = sl.split()
    # Multi-word: last word is a noise word → likely bled from section header
    if len(words) >= 2 and words[-1] in _NOISE_WORDS:
        return True
    return False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RoadmapNode:
    id:              str
    skill:           str
    priority:        str          # "high" | "medium" | "low"
    gap_score:       float
    week_start:      int
    week_end:        int
    duration_weeks:  float
    course_id:       Optional[str]
    course_title:    Optional[str]
    description:     str
    prerequisites:   list[str]    # skill names (already-known ones included for context)
    unlocked_by:     list[str]    # prerequisite node ids that must complete first
    category:        str
    is_prerequisite_only: bool    # True = pulled in as dep, not a direct gap


@dataclass
class RoadmapTimeline:
    candidate_name:   str
    target_role:      str
    match_score:      int
    total_weeks:      int
    high_weeks:       int
    medium_weeks:     int
    low_weeks:        int
    nodes:            list[RoadmapNode]
    known_skills:     list[str]
    summary:          str


# ── Skill taxonomy (category lookup) ─────────────────────────────────────────

_CATEGORY: dict[str, str] = {
    "Python": "Programming", "SQL": "Database", "Git": "Tools",
    "Docker": "DevOps", "Linux": "Tools", "AWS": "Cloud", "GCP": "Cloud",
    "Azure": "Cloud", "Kubernetes": "DevOps", "CI/CD": "DevOps",
    "Terraform": "DevOps", "GitHub Actions": "DevOps", "Jenkins": "DevOps",
    "SageMaker": "Cloud", "PyTorch": "ML Frameworks", "TensorFlow": "ML Frameworks",
    "Keras": "ML Frameworks", "Scikit-learn": "ML Libraries",
    "XGBoost": "ML Libraries", "NLP": "NLP", "BERT": "NLP",
    "Hugging Face": "NLP", "transformers": "NLP",
    "Feature Engineering": "ML", "MLflow": "MLOps", "Model Deployment": "MLOps",
    "Snowflake": "Data Warehouse", "BigQuery": "Data Warehouse",
    "dbt": "Data Engineering", "Kafka": "Data Streaming",
    "Spark": "Big Data", "Dask": "Big Data", "Airflow": "MLOps",
    "Feast": "MLOps", "Tecton": "MLOps", "feature store": "MLOps",
    "Pinecone": "Vector DB", "Weaviate": "Vector DB",
    "FAISS": "Vector DB", "LangChain": "AI Tools",
    "DCF": "Finance", "LBO": "Finance", "Financial Modelling": "Finance",
    "Credit Analysis": "Finance", "Bloomberg Terminal": "Finance",
    "Google Analytics": "Analytics", "SEO": "Marketing",
    "HubSpot": "CRM", "Salesforce": "CRM",
}


def _category(skill: str) -> str:
    return _CATEGORY.get(skill, "Technical")


# ── Priority from score ───────────────────────────────────────────────────────

def _priority(score: float) -> str:
    if score < THRESHOLD_HIGH:   return "high"
    if score < THRESHOLD_MEDIUM: return "medium"
    return "low"


# ── Core engine ───────────────────────────────────────────────────────────────

def build_roadmap(
    gap_report:       dict,
    matched_skills:   list[str],
    candidate_name:   str = "Candidate",
    target_role:      str = "Target Role",
    match_score:      int = 0,
    courses_json:     str = "data/courses.json",
) -> RoadmapTimeline:
    """
    Builds a personalised, dependency-ordered learning roadmap.

    Args:
        gap_report:     Output of gap_engine.run_gap_engine() — contains
                        gaps[], recommendations[], training_hints.
        matched_skills: Skills the candidate already has (from /analyze).
        candidate_name: Candidate's name for the timeline header.
        target_role:    JD target role title.
        match_score:    Overall match % from /analyze.
        courses_json:   Path to courses.json.

    Returns:
        RoadmapTimeline — fully ordered, week-stamped node list.
    """

    known_lower = {s.lower() for s in matched_skills}

    # ── Step 1: Load course catalog ───────────────────────────────────────────
    try:
        courses = json.loads(Path(courses_json).read_text())
        course_map = {c["id"]: c for c in courses}
    except Exception:
        course_map = {}

    # ── Step 2: Build recommendation lookup (gap_skill → course) ─────────────
    rec_map: dict[str, dict] = {}
    for rec in gap_report.get("recommendations", []):
        gs = rec["gap_skill"]
        if gs not in rec_map:
            rec_map[gs] = rec

    # ── Step 3: Filter gaps — remove noise, already-known ────────────────────
    raw_gaps   = gap_report.get("gaps", [])
    clean_gaps = []
    for g in raw_gaps:
        skill = g["skill"]
        if _is_noise(skill):
            continue
        if skill.lower() in known_lower:
            continue
        clean_gaps.append(g)

    # ── Step 4: Collect all skills needed (gaps + their prerequisites) ────────
    # Only pull in prerequisites that are NOT already known.
    required: dict[str, float] = {}   # skill → gap_score (0.0 for pure prereqs)
    is_direct_gap: set[str]    = set()

    for g in clean_gaps:
        skill = g["skill"]
        required[skill] = g["score"]
        is_direct_gap.add(skill)

    def _add_deps(skill: str, depth: int = 0):
        if depth > 8:
            return
        _, deps = SKILL_DEPS.get(skill, ("core", []))
        for dep in deps:
            if dep.lower() not in known_lower and dep not in required:
                required[dep] = 0.0   # prerequisite, no gap score
                _add_deps(dep, depth + 1)

    for g in clean_gaps:
        _add_deps(g["skill"])

    # ── Step 5: Topological sort (Kahn's algorithm) ───────────────────────────
    in_degree: dict[str, int]      = {s: 0 for s in required}
    dependents: dict[str, list[str]] = defaultdict(list)

    for skill in required:
        _, deps = SKILL_DEPS.get(skill, ("core", []))
        for dep in deps:
            if dep in required:
                in_degree[skill] += 1
                dependents[dep].append(skill)

    queue  = deque(s for s, d in in_degree.items() if d == 0)
    order  = []
    visited = set()

    while queue:
        # Among zero-in-degree nodes: sort by (priority desc, score asc)
        # so high-priority gaps surface before low-priority prerequisites
        candidates = sorted(
            queue,
            key=lambda s: (
                {"high": 0, "medium": 1, "low": 2}.get(
                    _priority(required.get(s, 0.5)), 2
                ),
                required.get(s, 0.5),
            )
        )
        skill = candidates[0]
        queue.remove(skill)

        if skill in visited:
            continue
        visited.add(skill)
        order.append(skill)

        for dep in dependents[skill]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    # Append any remaining (cycle guard)
    for skill in required:
        if skill not in visited:
            order.append(skill)

    # ── Step 6: Course grouping — merge same-course gaps into one node ────────
    # course_id → list of skills it covers among our gaps
    course_groups: dict[str, list[str]] = defaultdict(list)
    skill_to_course: dict[str, str]     = {}

    for skill in order:
        if skill not in is_direct_gap:
            continue
        rec = rec_map.get(skill)
        if rec:
            cid = rec["course_id"]
            course_groups[cid].append(skill)
            skill_to_course[skill] = cid

    # ── Step 7: Build nodes with week offsets ─────────────────────────────────
    nodes: list[RoadmapNode] = []
    skill_to_node_id: dict[str, str] = {}
    week_cursor = 1
    emitted_courses: set[str] = set()
    node_idx = 0

    for skill in order:
        if skill not in required:
            continue

        score    = required[skill]
        priority = _priority(score) if skill in is_direct_gap else "low"
        duration = DURATIONS.get(skill, 1.0)

        # Prerequisites list (all deps, mark known ones for context)
        _, raw_deps = SKILL_DEPS.get(skill, ("core", []))
        prereqs = raw_deps  # full list for display context

        # Unlocked_by: only deps that are in our roadmap (not already known)
        unlocked_by = [
            skill_to_node_id[dep]
            for dep in raw_deps
            if dep in required and dep in skill_to_node_id
        ]

        # Course assignment
        cid         = skill_to_course.get(skill)
        course_obj  = course_map.get(cid) if cid else None
        course_title = course_obj["title"] if course_obj else None

        # If this skill shares a course with a previous skill, skip a new node
        # — it's covered. But only skip direct gap skills (not prereqs).
        if skill in is_direct_gap and cid and cid in emitted_courses:
            # Still register the skill's node id pointing to the course node
            # (find which node has this course)
            for n in nodes:
                if n.course_id == cid:
                    skill_to_node_id[skill] = n.id
                    break
            continue

        node_id = f"node_{node_idx:02d}"
        node_idx += 1
        skill_to_node_id[skill] = node_id

        node = RoadmapNode(
            id               = node_id,
            skill            = skill,
            priority         = priority,
            gap_score        = round(score, 4),
            week_start       = week_cursor,
            week_end         = week_cursor + int(duration) - 1 or week_cursor,
            duration_weeks   = duration,
            course_id        = cid,
            course_title     = course_title,
            description      = DESCRIPTIONS.get(skill, f"Build practical skills in {skill}."),
            prerequisites    = prereqs,
            unlocked_by      = unlocked_by,
            category         = _category(skill),
            is_prerequisite_only = skill not in is_direct_gap,
        )
        nodes.append(node)
        week_cursor += max(1, int(duration))

        if cid:
            emitted_courses.add(cid)

    # ── Step 8: Timeline metrics ──────────────────────────────────────────────
    high_weeks   = sum(n.duration_weeks for n in nodes if n.priority == "high")
    medium_weeks = sum(n.duration_weeks for n in nodes if n.priority == "medium")
    low_weeks    = sum(n.duration_weeks for n in nodes if n.priority == "low")
    total_weeks  = week_cursor - 1

    n_gaps    = len(clean_gaps)
    n_high    = sum(1 for n in nodes if n.priority == "high" and not n.is_prerequisite_only)
    n_medium  = sum(1 for n in nodes if n.priority == "medium" and not n.is_prerequisite_only)

    summary = (
        f"{candidate_name} has {len(matched_skills)} matched skills and "
        f"{n_gaps} skill gaps for the {target_role} role. "
        f"{n_high} gaps are high priority (learn immediately), "
        f"{n_medium} are medium priority (bridge while applying). "
        f"Estimated completion: {total_weeks} weeks."
    )

    return RoadmapTimeline(
        candidate_name = candidate_name,
        target_role    = target_role,
        match_score    = match_score,
        total_weeks    = total_weeks,
        high_weeks     = int(high_weeks),
        medium_weeks   = int(medium_weeks),
        low_weeks      = int(low_weeks),
        nodes          = nodes,
        known_skills   = matched_skills,
        summary        = summary,
    )


def roadmap_to_dict(rt: RoadmapTimeline) -> dict:
    """Serialise RoadmapTimeline to a plain dict for JSON responses."""
    d = asdict(rt)
    return d


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="Build a skill gap roadmap")
    parser.add_argument("--gap_report",    default="data/gap_report.json")
    parser.add_argument("--matched",       default="Python,SQL,Docker,Git,AWS,Terraform,Scikit-learn")
    parser.add_argument("--name",          default="John Smith")
    parser.add_argument("--role",          default="ML Engineer")
    parser.add_argument("--score",         type=int, default=42)
    parser.add_argument("--courses",       default="data/courses.json")
    parser.add_argument("--output",        default=None)
    args = parser.parse_args()

    gap_report    = _json.loads(Path(args.gap_report).read_text())
    matched       = [s.strip() for s in args.matched.split(",") if s.strip()]

    timeline = build_roadmap(
        gap_report     = gap_report,
        matched_skills = matched,
        candidate_name = args.name,
        target_role    = args.role,
        match_score    = args.score,
        courses_json   = args.courses,
    )

    out = roadmap_to_dict(timeline)
    text = _json.dumps(out, indent=2)

    if args.output:
        Path(args.output).write_text(text)
        print(f"[OK] Roadmap written → {args.output}")
    else:
        print(text)
