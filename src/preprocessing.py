import re
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK assets
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

BASE_DIR = Path(__file__).resolve().parent.parent   # SmartBlogTagger/


class TextPreprocessor:

    def load_dataset(self, path: str):
        """Load CSV or JSON dataset with auto-detection."""
        full_path = BASE_DIR / path

        if not full_path.exists():
            raise FileNotFoundError(f"❌ Dataset not found at: {full_path}")

        print(f"📌 Loading dataset from: {full_path}")

        # Auto-detect format
        if str(full_path).endswith(".csv"):
            df = pd.read_csv(full_path)

        elif str(full_path).endswith(".json"):
            df = pd.read_json(full_path, lines=True)

        else:
            raise ValueError("Unsupported dataset format! Use CSV or JSON.")

        # Rename required columns
        rename_map = {
            "text": "content",
            "topic": "label",       # IMPORTANT: your dataset uses “topic”
            "target": "label"
        }
        df = df.rename(columns=rename_map)

        # Check required columns
        required_cols = ["content", "label"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"❌ Required column '{col}' is missing in dataset!")

        df = df.dropna(subset=["content", "label"]).reset_index(drop=True)
        return df

    def clean_text(self, text: str):
        """Clean a single text record."""
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"http\S+|www.\S+", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_stopwords_lemmatize(self, tokens):
        """Stopword removal + lemmatization."""
        stop_words = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
        return [lem.lemmatize(t) for t in tokens if t not in stop_words]

    def preprocess_column(self, df):
        """Apply full preprocessing pipeline."""
        print("🧹 Cleaning text...")
        df["clean_text"] = df["content"].apply(self.clean_text)

        print("🔎 Tokenizing...")
        df["tokens"] = df["clean_text"].apply(lambda x: x.split())

        print("🧽 Removing stopwords & lemmatizing...")
        df["tokens"] = df["tokens"].apply(self.remove_stopwords_lemmatize)

        print("🔗 Generating final processed text...")
        df["final_text"] = df["tokens"].apply(lambda x: " ".join(x))

        # return only final cleaned text + label
        return df[["final_text", "label"]]


if __name__ == "__main__":
    p = TextPreprocessor()
    df = p.load_dataset("data/blogtext.csv")   # CHANGE HERE
    clean_df = p.preprocess_column(df)
    print(clean_df.head())
