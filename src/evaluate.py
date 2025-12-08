import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import TextPreprocessor
from FeatureExtractor import FeatureExtractor

BASE_DIR = Path(__file__).resolve().parent.parent


class ModelEvaluator:

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.fe = FeatureExtractor()

        self.model = joblib.load(BASE_DIR / "outputs/models/classifier.pkl")
        self.label_encoder = joblib.load(BASE_DIR / "outputs/models/label_encoder.pkl")

        self.fe.load("outputs/models/features")

    def load_data(self):
        print("📌 Loading dataset for evaluation...")
        df = self.preprocessor.load_dataset("data/newsgroups.json")
        return df

    def evaluate(self):
        df = self.load_data()

        X = self.fe.transform(df["content"])
        y_true = self.label_encoder.transform(df["label"])

        print("📌 Making predictions...")
        y_pred = self.model.predict(X)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print("\n===== METRICS =====")
        print("Accuracy :", acc)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1-Score :", f1)

        # Save metrics report
        eval_dir = BASE_DIR / "outputs/evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        with open(eval_dir / "metrics.txt", "w") as f:
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1-Score: {f1}\n")
            f.write("\n\nFull Report:\n")
            f.write(classification_report(y_true, y_pred))

        # Generate Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        labels = self.label_encoder.classes_

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(eval_dir / "confusion_matrix.png")
        plt.close()

        # Prediction Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=y_pred)
        plt.title("Prediction Distribution")
        plt.xlabel("Predicted Labels")
        plt.ylabel("Count")
        plt.savefig(eval_dir / "prediction_distribution.png")
        plt.close()

        print("✅ Evaluation complete! Results saved to outputs/evaluation/")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate()
