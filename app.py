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
    print("Set HF_TOKEN in your terminal, for example:")

    print("Then run:")
    print("  python app/upload.py")
    raise SystemExit(0)

api = HfApi()
api.upload_folder(
    folder_path=repo_root,
    repo_id=repo_id,
    repo_type="model",
    token=hf_token,
)
print("Upload complete!")
