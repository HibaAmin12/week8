from pathlib import Path
import tiktoken


# ============================================================
# 1. LOAD DOCUMENT
# ============================================================

text = Path("../outputs/extracted_text.txt").read_text(
    encoding="utf-8"
)

tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode(text)


# ============================================================
# 2. SETTINGS
# ============================================================

CHUNK_SIZE = 400
OVERLAP = 60


# ============================================================
# 3. SHOW BOUNDARIES
# ============================================================

def show_boundary(start, label):

    end = start + CHUNK_SIZE

    before = tokenizer.decode(
        tokens[max(0, end - 80):end]
    )

    after = tokenizer.decode(
        tokens[end:end + 80]
    )

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    print("\n--- END OF CHUNK ---")
    print(before)

    print("\n--- START OF NEXT CHUNK ---")
    print(after)


# ============================================================
# 4. INSPECT SELECTED BOUNDARIES
# ============================================================

for chunk_number in [4, 9, 14, 19, 24, 29, 34]:

    start = chunk_number * CHUNK_SIZE

    print(
        f"\n\n{'#' * 80}"
    )

    print(
        f"BOUNDARY AFTER CHUNK {chunk_number}"
    )

    show_boundary(
        start,
        f"Boundary after chunk {chunk_number}"
    )
