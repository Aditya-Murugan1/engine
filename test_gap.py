from skill_extractor import extract_entities
from gap_engine import run_gap_engine
from pathlib import Path

resume = Path('data/sample_resume.txt').read_text()
jd     = Path('data/sample_jd.txt').read_text()

entities = extract_entities(resume, model_dir='models/skill_extractor')
print('Skills found:', entities['SKILL'])
print()

report = run_gap_engine(
    extracted_entities=entities,
    jd_text=jd,
    role_query='Machine Learning Engineer',
    onet_csv='data/onet_skills.csv',
    courses_json='data/courses.json',
    output_path='data/gap_report.json',
)

print('Gaps:')
for g in report['gaps'][:8]:
    print(f"  {g['skill']:<25} sim={g['score']}  [{g['source']}]")

print()
print('Course recommendations:')
for r in report['recommendations']:
    print(f"  {r['gap_skill']:<20} -> {r['course_title']}  ({r['match_score']})")

print()
print('Class weights:', report['training_hints']['class_weights'])
