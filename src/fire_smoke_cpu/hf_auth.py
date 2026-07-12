import os
from pathlib import Path

def load_dotenv(env_path=".env"):
    p = Path(env_path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')

def get_hf_token():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    return token if token else None

def require_hf_token():
    token = get_hf_token()
    if not token:
        raise ValueError("Hugging Face authentication required. Please set HF_TOKEN environment variable.")
    return token

def masked_token_status():
    token = get_hf_token()
    if not token:
        return {"exists": False, "preview": None}
    
    if len(token) > 4:
        preview = f"{token[:4]}..."
    else:
        preview = "***"
    return {"exists": True, "preview": preview}

def get_authenticated_hf_api():
    token = require_hf_token()
    try:
        from huggingface_hub import HfApi
        return HfApi(token=token)
    except ImportError:
        # Fallback to simple API access or raise if strictly required
        raise ImportError("huggingface_hub is not installed. Please install it to use get_authenticated_hf_api().")

def snapshot_download_safe(*args, **kwargs):
    token = get_hf_token()
    if token:
        kwargs["token"] = token
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(*args, **kwargs)
    except ImportError:
        raise ImportError("huggingface_hub is not installed. Please install it to use snapshot_download_safe().")
