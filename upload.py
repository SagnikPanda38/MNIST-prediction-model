import os
import sys
from huggingface_hub import HfApi


def get_hf_token():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    if len(sys.argv) > 1:
        return sys.argv[1]

    return None


repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_id = os.getenv("HF_REPO_ID", "DragonWarrior38/cnn_model")

hf_token = get_hf_token()
if not hf_token:
    print("No Hugging Face token was provided. Upload skipped.")
    print("Set HF_TOKEN in your terminal environment variables first.")
    print("Then run: python app/upload.py")
    raise SystemExit(0)

# =====================================================================
# 🛡️ FIXED: IGNORE HIDDEN DIRECTORIES TO PREVENT SYSTEM CACHE ERRORS
# =====================================================================
ignore_patterns = [
    "*/.cache/*",
    ".cache/*",
    "*/__pycache__/*",
    "__pycache__/*",
    "*/.git/*",
    ".git/*",
    "venv/*",
    "*/venv/*",
    "*.pyc"
]

print(f"Beginning synchronization for folder path: {repo_root}")
print(f"Target Space repository: {repo_id}")

api = HfApi()
api.upload_folder(
    folder_path=repo_root,
    repo_id=repo_id,
    repo_type="space",  
    token=hf_token,
    ignore_patterns=ignore_patterns, 
)

print("Upload complete! Check your Hugging Face Space page to view the active build.")
