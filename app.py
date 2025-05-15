# app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model from Hugging Face
MODEL_NAME = "zeeshanabbasi2004/finbert-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Sentiment prediction
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    labels = ['negative', 'neutral', 'positive']
    return labels[probs.argmax()], round(probs.max() * 100, 2)

# Streamlit UI
st.set_page_config(page_title="Financial Sentiment Analyzer", layout="centered")

st.title("💰 Financial News Sentiment Analyzer")
st.markdown("Enter a financial headline to classify its sentiment:")

headline = st.text_input("🔍 Enter Headline:", "")

if st.button("Analyze"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        sentiment, confidence = predict_sentiment(headline)
        st.success(f"**Sentiment:** {sentiment.upper()} ({confidence}%)")