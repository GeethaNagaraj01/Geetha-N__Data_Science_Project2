import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


BASE_DIR = Path(__file__).resolve().parent.parent


class FeatureExtractor:

    def __init__(self, max_features=5000, n_components=150):
        """
        TF-IDF + SVD Dimensionality Reduction
        Faster & more compact than your previous version.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features
        )
        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )

    def fit_transform(self, texts):
        """
        Fit vectorizer + SVD, return reduced embeddings.
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        reduced_matrix = self.svd.fit_transform(tfidf_matrix)
        return reduced_matrix

    def transform(self, texts):
        """
        Transform new text using already-fitted vectorizer + SVD.
        """
        tfidf_matrix = self.vectorizer.transform(texts)
        reduced_matrix = self.svd.transform(tfidf_matrix)
        return reduced_matrix

    def save(self, path="outputs/models/features"):
        """
        Save vectorizer + SVD to disk.
        """
        path = BASE_DIR / path
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vectorizer, path / "tfidf.pkl")
        joblib.dump(self.svd, path / "svd.pkl")

    def load(self, path="outputs/models/features"):
        """
        Load vectorizer + SVD from disk.
        """
        path = BASE_DIR / path
        self.vectorizer = joblib.load(path / "tfidf.pkl")
        self.svd = joblib.load(path / "svd.pkl")
