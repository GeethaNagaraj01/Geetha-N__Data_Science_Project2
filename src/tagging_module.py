import joblib
import numpy as np
from pathlib import Path
from .preprocessing import TextPreprocessor
from .FeatureExtractor import FeatureExtractor


BASE_DIR = Path(__file__).resolve().parent.parent


class SmartTagger:

    def __init__(self):
        # Load preprocessing pipeline
        self.pre = TextPreprocessor()

        # Load vectorizers & model
        self.fe = FeatureExtractor()
        self.fe.load("outputs/models/features")

        self.model = joblib.load(BASE_DIR / "outputs/models/classifier.pkl")
        self.label_encoder = joblib.load(BASE_DIR / "outputs/models/label_encoder.pkl")

        self.labels = list(self.label_encoder.classes_)

    def predict_tags(self, text):
        """
        Full tagging pipeline:
        1. Clean text
        2. Extract TF-IDF + SVD features
        3. Predict using trained model
        4. Return confidence & recommended tags
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {"error": "Empty text input."}

        # Step 1: Clean text
        cleaned = self.pre.clean_text(text)

        # Step 2: Feature extraction
        vector = self.fe.transform([cleaned])

        # Step 3: Probability prediction
        probabilities = self.model.predict_proba(vector)[0]

        # Step 4: Highest probability label
        top_idx = np.argmax(probabilities)
        top_tag = self.labels[top_idx]
        top_conf = float(probabilities[top_idx])

        # Step 5: Multi-tag recommendation
        threshold = 0.15  # tags with >15% probability are recommended
        recommended_tags = [
            {"tag": self.labels[i], "score": float(probabilities[i])}
            for i in range(len(probabilities))
            if probabilities[i] >= threshold
        ]

        # Sort by confidence
        recommended_tags = sorted(recommended_tags, key=lambda x: x["score"], reverse=True)

        # Step 6: Output dictionary
        return {
            "input_text": text,
            "cleaned_text": cleaned,
            "predicted_tag": top_tag,
            "confidence": round(top_conf, 4),
            "recommended_tags": recommended_tags,
            "probabilities": {
                self.labels[i]: float(probabilities[i]) for i in range(len(probabilities))
            }
        }


# Testing module directly
if __name__ == "__main__":
    tagger = SmartTagger()

    sample_text = """Apple releases new update with improved security features and new hardware support."""

    result = tagger.predict_tags(sample_text)
    print(result)
