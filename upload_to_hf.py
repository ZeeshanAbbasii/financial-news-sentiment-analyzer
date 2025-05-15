from huggingface_hub import HfApi, upload_folder

# Define your Hugging Face username and model repo name
username = "zeeshanabbasi2004"
repo_name = "finbert-sentiment"
repo_id = f"{username}/{repo_name}"

# Step 1: Create repo (only once)
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Step 2: Upload folder with model files
upload_folder(
    repo_id=repo_id,
    folder_path="models/finbert_model",  # ✅ Your correct model folder
    path_in_repo="",                     # Upload to root of the repo
    repo_type="model"
)