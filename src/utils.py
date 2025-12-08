import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# -----------------------
# CLEAN TEXT FUNCTION
# -----------------------
def clean_text(text: str):
    """
    Basic text cleaning for blog content
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)         # Remove links
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)        # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()           # Remove extra spaces
    return text


# -----------------------
# LOAD DATASET FUNCTION
# -----------------------
def load_dataset(filename="dataset.json"):
    """
    Load dataset from the data/ folder
    """
    path = BASE_DIR / "data" / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_json(path)
    df = df.rename(columns={"text": "content", "target": "label"})
    df = df.dropna().reset_index(drop=True)
    return df
