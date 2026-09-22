import pymupdf
from clean_text import clean_text


PDF_PATH = "Lost in the Middle.pdf"
OUTPUT_PATH = "outputs/extracted_text.txt"

doc = pymupdf.open(PDF_PATH)

full_text = ""

for page in doc:
    full_text += page.get_text("text") + "\n"

cleaned_text = clean_text(full_text)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write(cleaned_text)

print(f"Raw characters: {len(full_text)}")
print(f"Cleaned characters: {len(cleaned_text)}")
print(f"Saved cleaned text to: {OUTPUT_PATH}")