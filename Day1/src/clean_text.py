import re


def clean_text(text):
    # Fix words broken at the end of a line:
    # "abil-\nity" -> "ability"
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Convert line breaks into spaces
    text = text.replace("\n", " ")

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()