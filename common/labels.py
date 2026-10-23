"""Label lists (labels.txt, one name per line), shared by every recogniser's collect step."""
import re

# Names become folder names (items) and CSV values (gestures): keep them to letters, digits and _.
VALID_NAME = r"[a-z0-9_]+"


def load_labels(path):
    """All names in a labels file (blank lines ignored). Empty list if the file doesn't exist."""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def add_label(path, name, labels):
    """Add a new name to the labels file (and the list in memory)."""
    text = path.read_text() if path.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + name + "\n")
    labels.append(name)
    print(f"Added '{name}' to {path.name}")


def choose_label(path, labels):
    """Ask in the terminal which label to record. A new name is added to the labels file."""
    print(f"Labels in {path.name}:")
    for number, name in enumerate(labels, start=1):
        print(f"  {number}. {name}")
    while True:
        answer = input("Type a number, or a new name: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(labels):
            return labels[int(answer) - 1]
        if answer and not answer.isdigit():
            name = answer.lower().replace(" ", "_")
            if not re.fullmatch(VALID_NAME, name):
                print("Names can only use letters a-z, digits and _ (spaces become _).")
                continue
            if name not in labels:
                add_label(path, name, labels)
            return name
        print("Please type one of the numbers above, or a name.")
