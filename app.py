import streamlit as st
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# -----------------------------
# NER + Sentiment Module
# -----------------------------
class NERSentimentModule:
    def __init__(self):
        st.info("Loading NLP models... This may take a few seconds.")
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def extract_entities(self, text):
        """
        Extract named entities using spaCy.
        Returns list of dictionaries [{'text': entity, 'label': label}, ...]
        """
        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return entities

    def analyze_sentiment(self, text):
        """
        Perform sentiment analysis using VADER.
        Returns: Positive / Neutral / Negative
        """
        score = self.sentiment_analyzer.polarity_scores(text)
        compound = score["compound"]

        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def suggest_tags(self, text, top_n=5):
        """
        Generate recommended tags from entities and frequent nouns.
        """
        doc = self.nlp(text)
        # Get nouns and proper nouns
        nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        # Get named entities
        entities = [ent.text for ent in doc.ents]
        # Combine and count frequency
        all_terms = nouns + entities
        most_common = Counter(all_terms).most_common(top_n)
        tags = [term for term, _ in most_common]
        return tags

    def process_text(self, text):
        """
        Returns dictionary with entities, sentiment, and suggested tags.
        """
        entities = self.extract_entities(text)
        sentiment = self.analyze_sentiment(text)
        tags = self.suggest_tags(text)
        return {"entities": entities, "sentiment": sentiment, "tags": tags}


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Smart Blog Intelligence System", layout="wide")
st.title("🧠 Smart Blog Intelligence System")

blog_text = st.text_area("📝 Enter Blog Content", height=200)

if st.button("Analyze"):
    if blog_text.strip():
        module = NERSentimentModule()
        result = module.process_text(blog_text)

        # Display entities
        entities = result.get("entities", [])
        if entities:
            st.subheader("🔹 Extracted Entities")
            for ent in entities:
                st.write(f"- {ent['text']} ({ent['label']})")
        else:
            st.write("No entities found.")

        # Display sentiment
        sentiment = result.get("sentiment", "Neutral")
        st.subheader("🔹 Sentiment Analysis")
        st.write(sentiment)

        # Display suggested tags
        tags = result.get("tags", [])
        st.subheader("🔹 Suggested Tags")
        st.write(", ".join(tags) if tags else "No tags suggested.")

    else:
        st.warning("Please enter some blog content!")
