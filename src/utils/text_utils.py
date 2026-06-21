import string

# Vocabulary
VOCAB = list("abcdefghijklmnopqrstuvwxyz '")

char_to_idx = {
    char: idx + 1
    for idx, char in enumerate(VOCAB)
}

idx_to_char = {
    idx: char
    for char, idx in char_to_idx.items()
}

BLANK_TOKEN = 0


def parse_alignment(alignment_path):
    words = []

    with open(alignment_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 3:
            continue

        word = parts[2]

        if word != "sil":
            words.append(word)

    return " ".join(words)


def text_to_int(text):
    return [
        char_to_idx[c]
        for c in text.lower()
        if c in char_to_idx
    ]


def int_to_text(indices):
    return "".join([
        idx_to_char[i]
        for i in indices
        if i in idx_to_char
    ])