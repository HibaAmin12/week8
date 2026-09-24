from pathlib import Path
import json
import urllib.request


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "train-v1.1.json"
OUTPUT_PATH = BASE_DIR / "outputs" / "clean_text.txt"

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"


def download_squad():
    """Download the SQuAD training dataset if it does not exist."""

    if not DATA_PATH.exists():
        print("Downloading SQuAD dataset...")
        urllib.request.urlretrieve(SQUAD_URL, DATA_PATH)
        print(f"Downloaded to: {DATA_PATH}")
    else:
        print("SQuAD dataset already exists.")


def load_squad_text(num_samples=20):
    """
    Extract unique context paragraphs from SQuAD.
    """

    with DATA_PATH.open("r", encoding="utf-8") as f:
        squad_data = json.load(f)

    contexts = []

    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()

            if context:
                contexts.append(context)

            if len(contexts) >= num_samples:
                break

        if len(contexts) >= num_samples:
            break

    # Remove duplicates while preserving order
    unique_contexts = list(dict.fromkeys(contexts))

    text_data = "\n\n".join(unique_contexts)

    return text_data


if __name__ == "__main__":
    download_squad()

    text = load_squad_text(num_samples=20)

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    print(f"Total contexts: 20")
    print(f"Total characters: {len(text)}")
    print(f"Total words: {len(text.split())}")