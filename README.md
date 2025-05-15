# Financial News Sentiment Analyzer

This project uses a fine-tuned [FinBERT](https://huggingface.co/zeeshanabbasi2004/finbert-sentiment) transformer model to classify financial news headlines into **positive**, **neutral**, or **negative** sentiment.

The model is trained on the Financial PhraseBank dataset and integrates seamlessly with Hugging Face Transformers for easy inference and deployment.

---

## 🧠 Features

- Fine-tuned FinBERT model for financial sentiment analysis
- Loaded directly from Hugging Face (no local model files needed)
- Full preprocessing: stopword removal, lemmatization, tokenization
- Handles class imbalance via oversampling
- Traditional model benchmarking (Logistic Regression, SVM, XGBoost)
- Live sentiment prediction from terminal using `main.py`
- Clean and Colab-friendly Jupyter Notebook
- Deployment-ready with minimal setup

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ZeeshanAbbasii/financial-news-sentiment-analyzer.git
cd financial-news-sentiment-analyzer

2. Create and activate a virtual environment

python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

🔍 How to Use

✅ Run sentiment prediction from terminal

python main.py "Tesla shares rally after record-breaking Q2 earnings"

Example Output:

Headline: Tesla shares rally after record-breaking Q2 earnings
Predicted Sentiment: POSITIVE (94.21%)

📓 Run and explore the notebook

Open the notebook in Jupyter:

Navigate to notebooks/final_notebook.ipynb to:
	•	Explore EDA
	•	See model comparisons
	•	Test the FinBERT classifier on financial headlines

⸻

🤗 Model Access

The fine-tuned FinBERT model is publicly available on Hugging Face:

🔗 https://huggingface.co/zeeshanabbasi2004/finbert-sentiment

You can load it directly in your code:

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("zeeshanabbasi2004/finbert-sentiment")
tokenizer = AutoTokenizer.from_pretrained("zeeshanabbasi2004/finbert-sentiment")

No need to store .safetensors locally — it works out of the box.

⸻

🛠 Requirements

All dependencies are listed in requirements.txt. Key libraries include:
	•	transformers
	•	torch
	•	scikit-learn
	•	nltk
	•	xgboost
	•	wordcloud

To install:

pip install -r requirements.txt

🧑‍💻 Developer Notes
	•	Model trained and evaluated using a balanced version of Financial PhraseBank
	•	Class imbalance handled using random oversampling
	•	MPS (Metal GPU) supported for Apple Silicon devices
	•	Model supports fast inference and is ready for Streamlit or API deployment

⸻

📜 License

This project is licensed under the MIT License.

⸻

🙌 Credits
	•	FinBERT Model: yiyanghkust/finbert-tone
	•	Transformers Library: Hugging Face
	•	Created & Maintained By: Zeeshan Abbasi

⸻

💬 Contact

Feel free to reach out for collaboration, questions, or suggestions via GitHub or LinkedIn.