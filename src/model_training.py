import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing import TextPreprocessor
from FeatureExtractor import FeatureExtractor

BASE_DIR = Path(__file__).resolve().parent.parent


class ModelTrainer:

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.fe = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.model = OneVsRestClassifier(
            LogisticRegression(max_iter=2000)
        )

    # Load CSV dataset
    def load_data(self):
        print("📌 Loading dataset...")
        df = self.preprocessor.load_dataset("data/blogtext.csv")   # will read CSV
        print("📌 Dataset loaded:", df.shape)

        # rename columns to what model expects
        df = df.rename(columns={
            "text": "content",   # main text field
            "topic": "label"     # classification label
        })

        # check if necessary columns exist
        required = ["content", "label"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"❌ Missing required column: {col}")

        return df

    def encode_labels(self, labels):
        print("📌 Encoding labels...")
        y = self.label_encoder.fit_transform(labels)
        return y

    def train(self):
        df = self.load_data()

        # Encode labels
        y = self.encode_labels(df["label"])

        # Extract features
        print("📌 Extracting text features...")
        X = self.fe.fit_transform(df["content"])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("📌 Training model...")
        self.model.fit(X_train, y_train)

        print("📌 Evaluating model...")
        y_pred = self.model.predict(X_test)
        print("\n===== MODEL REPORT =====\n")
        print(classification_report(y_test, y_pred))

        self.save_all()
        print("✅ Training complete. Model saved successfully!")

    def save_all(self):
        model_dir = BASE_DIR / "outputs/models"
        model_dir.mkdir(parents=True, exist_ok=True)

        print("📌 Saving model and encoders...")

        joblib.dump(self.model, model_dir / "classifier.pkl")
        joblib.dump(self.label_encoder, model_dir / "label_encoder.pkl")
        self.fe.save("outputs/models/features")

        print("✅ All components saved!")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
