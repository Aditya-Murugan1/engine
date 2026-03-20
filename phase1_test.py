#!/usr/bin/env python3
# phase1_test.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Basic end-to-end pipeline test using a sample resume.
# No training data needed. Uses dslim/bert-base-NER directly for extraction.
#
# What this tests:
#   [1] Package imports are all working
#   [2] skill_extractor.extract_entities() runs on sample resume
#   [3] gap_engine.run_gap_engine() produces a valid gap report
#   [4] gap_report.json is written correctly
#   [5] train.py can load class weights from the gap report
#   [6] All data files (onet, courses) are readable and valid
#
# Run from inside the engine/ folder:
#   python phase1_test.py
#
# Expected output: PASS on all 6 checks + printed gap report summary
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import json
import traceback
from pathlib import Path

# Ensure engine/ is on sys.path so sibling modules (skill_extractor, gap_engine, train)
# are importable on Windows where the script directory isn't added automatically.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Colour helpers ────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗ FAIL{RESET}  {msg}")
def info(msg):  print(f"  {YELLOW}→{RESET} {msg}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}\n{'─'*60}")

results = []

def check(label: str, fn):
    """Runs fn(), records pass/fail, returns the return value or None."""
    try:
        val = fn()
        ok(label)
        results.append((label, True))
        return val
    except Exception as e:
        fail(f"{label}\n         {RED}{type(e).__name__}: {e}{RESET}")
        traceback.print_exc()
        results.append((label, False))
        return None

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Package imports
# ─────────────────────────────────────────────────────────────────────────────

header("CHECK 1 — Package imports")

def import_packages():
    import torch
    import transformers
    import sentence_transformers
    import sklearn
    import pandas
    import numpy
    import seqeval
    info(f"torch                 {torch.__version__}")
    info(f"transformers          {transformers.__version__}")
    info(f"sentence-transformers {sentence_transformers.__version__}")
    info(f"CUDA available        {torch.cuda.is_available()}")
    return True

check("All required packages importable", import_packages)

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Data files exist and are valid
# ─────────────────────────────────────────────────────────────────────────────

header("CHECK 2 — Data files")

def check_resume():
    text = Path("data/sample_resume.txt").read_text()
    assert len(text) > 100, "resume too short"
    info(f"Resume length: {len(text)} chars")
    return text

def check_jd():
    text = Path("data/sample_jd.txt").read_text()
    assert len(text) > 100, "JD too short"
    info(f"JD length: {len(text)} chars")
    return text

def check_onet():
    import pandas as pd
    # Person B delivers Skills.xlsx (62580 skills)
    if Path("data/Skills.xlsx").exists():
        df = pd.read_excel("data/Skills.xlsx")
        col = next((c for c in df.columns if "element" in c.lower() or "skill" in c.lower()), df.columns[1])
        info(f"Skills.xlsx rows: {len(df)} | sample col: '{col}'")
        return df
    elif Path("data/onet_skills.csv").exists():
        df = pd.read_csv("data/onet_skills.csv")
        assert "Title" in df.columns and "Element Name" in df.columns
        info(f"O*NET rows: {len(df)} | roles: {df['Title'].nunique()}")
        return df
    else:
        raise FileNotFoundError("Neither data/Skills.xlsx nor data/onet_skills.csv found")

def check_courses():
    courses = json.loads(Path("data/courses.json").read_text())
    assert isinstance(courses, list) and len(courses) > 0
    assert all("id" in c and "title" in c and "skills_covered" in c for c in courses)
    info(f"Courses loaded: {len(courses)}")
    return courses

resume_text = check("data/sample_resume.txt readable", check_resume)
jd_text     = check("data/sample_jd.txt readable",     check_jd)
check("data/Skills.xlsx or onet_skills.csv valid", check_onet)
check("data/courses.json valid",     check_courses)

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Entity extraction from sample resume
# ─────────────────────────────────────────────────────────────────────────────

header("CHECK 3 — Entity extraction (dslim/bert-base-NER)")
info("Downloading dslim/bert-base-NER if not cached — first run may take ~1 min")

entities = None

def run_extraction():
    from skill_extractor import extract_entities
    global entities
    # Phase 1: uses BASE_MODEL (no fine-tuning yet)
    entities = extract_entities(resume_text)
    assert isinstance(entities, dict)
    assert "SKILL" in entities and "EXPERIENCE" in entities and "EDUCATION" in entities
    total = sum(len(v) for v in entities.values())
    assert total > 0, "No entities extracted at all"
    return entities

entities = check("extract_entities() runs and returns dict", run_extraction)

if entities:
    header("  Extracted entities")
    for etype, items in entities.items():
        info(f"{etype:<12} ({len(items)}) : {items[:8]}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Gap engine produces valid report
# ─────────────────────────────────────────────────────────────────────────────

header("CHECK 4 — Gap engine")

gap_report = None

def run_gap_engine():
    global gap_report
    if entities is None:
        raise RuntimeError("Skipping — entity extraction failed")

    from gap_engine import run_gap_engine as _run
    onet_path = "data/Skills.xlsx" if Path("data/Skills.xlsx").exists() else "data/onet_skills.csv"
    gap_report = _run(
        extracted_entities=entities,
        jd_text=jd_text,
        role_query="Machine Learning Engineer",
        onet_csv=onet_path,
        courses_json="data/courses.json",
        output_path="data/gap_report.json",
    )
    assert "gaps" in gap_report
    assert "recommendations" in gap_report
    assert "training_hints" in gap_report
    assert "class_weights" in gap_report["training_hints"]
    return gap_report

gap_report = check("run_gap_engine() produces valid report", run_gap_engine)

if gap_report:
    header("  Gap report summary")
    gaps  = gap_report["gaps"]
    recs  = gap_report["recommendations"]
    hints = gap_report["training_hints"]["class_weights"]

    info(f"Total gaps found      : {len(gaps)}")
    info(f"Course recommendations: {len(recs)}")
    info(f"Class weights         : {hints}")

    print(f"\n  {BOLD}Top 5 skill gaps:{RESET}")
    for g in gaps[:5]:
        print(f"    {'▸'} {g['skill']:<30} sim={g['score']}  [{g['source']}]")

    print(f"\n  {BOLD}Top 5 course recommendations:{RESET}")
    for r in recs[:5]:
        print(f"    {'▸'} [{r['gap_skill']:<20}] → {r['course_title'][:45]}  (score={r['match_score']})")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — gap_report.json written to disk correctly
# ─────────────────────────────────────────────────────────────────────────────

header("CHECK 5 — gap_report.json on disk")

def check_gap_report_file():
    path   = Path("data/gap_report.json")
    assert path.exists(), "File not created"
    loaded = json.loads(path.read_text())
    assert loaded["gaps"] is not None
    info(f"File size: {path.stat().st_size} bytes")
    return True

check("data/gap_report.json written and readable", check_gap_report_file)

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — train.py can load class weights from report
# ─────────────────────────────────────────────────────────────────────────────

header("CHECK 6 — Training launcher weight loader")

def check_weight_loader():
    import torch
    from train import load_class_weights, LABELS
    weights = load_class_weights("data/gap_report.json")
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == len(LABELS)
    assert all(w > 0 for w in weights.tolist())
    info(f"Weight tensor shape : {list(weights.shape)}")
    info(f"Weight values       : {[round(w, 3) for w in weights.tolist()]}")
    return weights

check("load_class_weights() reads gap_report and returns valid tensor", check_weight_loader)

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

header("PHASE 1 SUMMARY")

passed = sum(1 for _, r in results if r)
total  = len(results)

for label, result in results:
    icon = f"{GREEN}✓{RESET}" if result else f"{RED}✗{RESET}"
    print(f"  {icon}  {label}")

print(f"\n  {BOLD}Result: {passed}/{total} checks passed{RESET}")

if passed == total:
    print(f"\n  {GREEN}{BOLD}✓ Phase 1 complete. Pipeline is end-to-end functional.{RESET}")
    print(f"  {YELLOW}Next step: hand Person B the data format spec and collect train.jsonl / val.jsonl.{RESET}")
    print(f"  {YELLOW}Then run: python train.py{RESET}\n")
else:
    failed = [l for l, r in results if not r]
    print(f"\n  {RED}{BOLD}✗ {total - passed} check(s) failed:{RESET}")
    for l in failed:
        print(f"    - {l}")
    print(f"\n  Fix the above before moving to phase 2.\n")
    sys.exit(1)
