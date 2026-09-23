import tiktoken

TEXT_PATH = "../outputs/extracted_text.txt"

with open(TEXT_PATH, "r", encoding="utf-8") as file:
    text = file.read()

encoding = tiktoken.get_encoding("cl100k_base")

tokens = encoding.encode(text)

print(f"Characters: {len(text)}")
print(f"Words: {len(text.split())}")
print(f"Tokens: {len(tokens)}")