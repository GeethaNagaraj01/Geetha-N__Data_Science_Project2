import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NERSentimentModule:
    def __init__(self):
        print("Loading spaCy model for NER...")
        self.nlp = spacy.load("en_core_web_sm")

        print("Loading VADER sentiment analyzer...")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def extract_entities(self, text):
        """
        Extract named entities using spaCy.
        Returns a list of dictionaries: [{'text': entity, 'label': label}, ...]
        """
        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return entities

    def analyze_sentiment(self, text):
        """
        Perform sentiment analysis using VADER.
        Returns sentiment label: Positive / Neutral / Negative
        """
        score = self.sentiment_analyzer.polarity_scores(text)
        compound = score["compound"]

        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def process_text(self, text):
        """
        Combined method for Streamlit compatibility:
        Returns a dictionary with keys 'entities' and 'sentiment'.
        This prevents tuple-related errors.
        """
        ents = self.extract_entities(text)
        sentiment = self.analyze_sentiment(text)
        return {"entities": ents, "sentiment": sentiment}


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    demo_text = """
    Apple has released the latest version of iOS with major security improvements 
    and enhanced privacy features. The update also brings support for new hardware 
    devices and fixes several bugs reported by users in the previous version. 
    Analysts predict strong adoption among iPhone users in the coming months.
    """

    module = NERSentimentModule()
    ner_output = module.process_text(demo_text)

    print("\n======== NER RESULTS ========")
    for ent in ner_output["entities"]:
        print(ent)

    print("\n======== SENTIMENT RESULTS ========")
    print(ner_output["sentiment"])
