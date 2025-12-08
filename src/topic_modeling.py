import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import NMF
from .preprocessing import TextPreprocessor
from .FeatureExtractor import FeatureExtractor

BASE_DIR = Path(__file__).resolve().parent.parent


class TopicModeling:

    def __init__(self, n_topics=10, top_words=10):
        self.n_topics = n_topics
        self.top_words = top_words

        # Initialize preprocessing and feature extraction
        self.pre = TextPreprocessor()
        self.fe = FeatureExtractor()
        self.fe.load(BASE_DIR / "outputs/models/features")  # Load TF-IDF + SVD

        # Placeholder for trained NMF model
        self.nmf = None
        self.H = None

    # Fit NMF on a dataset
    def fit_topics(self, df):
        print("Cleaning dataset...")
        df["cleaned"] = df["content"].apply(self.pre.clean_text)

        print("Extracting TF-IDF matrix (non-negative)...")
        X = self.fe.vectorizer.transform(df["cleaned"])

        print("Running NMF topic model...")
        self.nmf = NMF(
            n_components=self.n_topics,
            init='nndsvd',
            random_state=42,
            max_iter=400
        )

        W = self.nmf.fit_transform(X)
        self.H = self.nmf.components_

        feature_names = self.fe.vectorizer.get_feature_names_out()

        topics = []
        for topic_idx, topic_vec in enumerate(self.H):
            top_indices = topic_vec.argsort()[-self.top_words:][::-1]
            words = [feature_names[i] for i in top_indices]

            topics.append({
                "topic_id": int(topic_idx),
                "keywords": words
            })

        # Save topics to JSON
        out_path = BASE_DIR / "outputs/topics.json"
        with open(out_path, "w") as f:
            json.dump(topics, f, indent=4)

        print(f"Saved topics at {out_path}")
        return topics, W

    # Predict topic for a single text
    def predict_topic(self, text):
        cleaned = self.pre.clean_text(text)
        X = self.fe.vectorizer.transform([cleaned])

        if self.nmf is not None and self.H is not None:
            # Use NMF model if trained
            W_single = self.nmf.transform(X)
            top_topic_idx = np.argmax(W_single)
            top_keywords = [self.fe.vectorizer.get_feature_names_out()[i]
                            for i in self.H[top_topic_idx].argsort()[-self.top_words:][::-1]]
            topics = [f"Topic {top_topic_idx}: {', '.join(top_keywords)}"]
            W = W_single
        else:
            # Dummy topics if NMF not trained
            topics = ["SampleTopic1", "SampleTopic2"]
            W = None

        return topics, W

    # Print topics nicely
    def print_topics(self, topics):
        for t in topics:
            print(f"Topic {t['topic_id']}: {', '.join(t['keywords'])}")


# Example usage
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_json(BASE_DIR / "data/newsgroups.json")
    df = df.rename(columns={"text": "content", "target": "label"})
    df = df.dropna().reset_index(drop=True)

    tm = TopicModeling(n_topics=12, top_words=12)
    topics, W = tm.fit_topics(df)
    tm.print_topics(topics)
    print("Topic modeling completed!")

    # Test single text prediction
    sample_text = "Apple releases a new iOS update with security improvements."
    pred_topics, _ = tm.predict_topic(sample_text)
    print("Predicted topic for sample text:", pred_topics)
