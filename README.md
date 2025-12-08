# Geetha-N__Data_Science_Project2

Smart Blog Intelligence System
Overview

Smart Blog Intelligence System is an interactive NLP-powered tool that automatically extracts key entities, performs sentiment analysis, and generates suggested tags from blog content.
This project is an enhanced version of the original Automated Content Tagging System for Blogs, developed based on feedback to include advanced NLP techniques, performance metrics, and deployment readiness.

Features

Named Entity Recognition (NER): Detects organizations, products, dates, and other key entities using spaCy.

Sentiment Analysis: Classifies blog content as Positive, Neutral, or Negative using VADER.

Suggested Tags: Generates SEO-friendly tags by combining named entities and frequent nouns.

Interactive Interface: Streamlit app for real-time analysis.

Performance Metrics: Includes accuracy, precision, recall, F1-score, and confusion matrix for NLP models.

Advanced NLP Techniques: Lemmatization, POS tagging, enhanced NER, and keyword extraction (RAKE).

Deployment Ready: Fully functional Streamlit app with visualizations and analysis.

Project Structure
Smart-Blog-Intelligence-System/
│
├── src/                        # Source code modules
│   ├── ner_sentiment_module.py # NER and sentiment analysis
│   ├── tagging_module.py       # Tag generation module
│   └── utils.py                # Utility functions
│
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── dataset/                    # Sample dataset or data link
├── README.md                   # Project documentation
└── assets/                     # Screenshots or images (optional)

Installation

Clone the repository:

git clone https://github.com/your-username/Smart-Blog-Intelligence-System.git
cd Smart-Blog-Intelligence-System


Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows


Install dependencies:

pip install -r requirements.txt

Usage

Run the Streamlit app:

streamlit run app.py


Enter blog content in the text area.

Click Analyze to get:

Named Entities

Sentiment (Positive / Neutral / Negative)

Suggested Tags

Dataset

A sample dataset is provided in the dataset/ folder.

For full-scale testing, you may use your own blog content or publicly available blog datasets.

(Optional: Provide Kaggle dataset link if applicable)

Results

Extracted entities are displayed with labels.

Sentiment is shown as Positive, Neutral, or Negative.

Suggested tags are generated automatically from entities and frequent nouns.

Visualizations and analysis are included for better understanding.

Future Improvements

Integrate more advanced NER models (e.g., Hugging Face transformers).

Improve tag generation using semantic similarity and keyword ranking.

Add user authentication and cloud deployment.

Screenshots

<img width="1847" height="1105" alt="Screenshot from 2025-12-08 21-11-25" src="https://github.com/user-attachments/assets/2de9c877-48c2-4f51-9c38-739bf378678b" />
