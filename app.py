# app.py

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT model from Hugging Face
MODEL_NAME = "zeeshanabbasi2004/finbert-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define prediction function
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    labels = ['Negative', 'Neutral', 'Positive']
    prediction = labels[probs.argmax()]
    confidence = round(probs.max() * 100, 2)
    
    return f"{prediction} ({confidence}%)"

# Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter financial headline here..."),
    outputs="text",
    title="Financial News Sentiment Analyzer",
    description="Classify financial headlines as Positive, Neutral, or Negative using FinBERT"
)

if __name__ == "__main__":
    demo.launch()
