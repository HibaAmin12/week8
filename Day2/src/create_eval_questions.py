import json
from pathlib import Path

DATA_PATH = Path("Day2/data/train-v1.1.json")
OUTPUT_PATH = Path("Day2/outputs/eval_questions.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = []

for article in data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]

        for qa in paragraph["qas"]:
            questions.append({
                "question": qa["question"],
                "answer": qa["answers"][0]["text"],
                "context": context
            })

# Keep a small, manageable evaluation set
questions = questions[:20]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"Created {len(questions)} evaluation questions.")
print(f"Saved to: {OUTPUT_PATH}")