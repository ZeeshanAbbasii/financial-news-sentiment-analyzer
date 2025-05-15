import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# ✅ Load model and tokenizer directly from Hugging Face
MODEL_NAME = "zeeshanabbasi2004/finbert-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ✅ Use MPS if available (Mac), else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# ✅ Sentiment prediction function
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    labels = ['negative', 'neutral', 'positive']
    predicted_label = labels[probs.argmax()]
    confidence = round(probs.max() * 100, 2)

    return predicted_label, confidence

# ✅ Run from terminal
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your financial headline here\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    label, score = predict_sentiment(input_text)

    print(f"\nHeadline: {input_text}")
    print(f"Predicted Sentiment: {label.upper()} ({score}%)")