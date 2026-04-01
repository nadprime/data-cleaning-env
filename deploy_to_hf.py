"""
deploy_to_hf.py
===============
Helper script to deploy this environment to HuggingFace Spaces.

What it does:
  1. Creates a Docker-based HuggingFace Space if it doesn't exist
  2. Uploads all project files to the Space
  3. Prints the Space URL where your environment will be live

After running, HF Spaces will build the Docker image (takes 2–5 minutes).
Monitor the build at: https://huggingface.co/spaces/YOUR_USERNAME/data-cleaning-env
"""
import os
import sys

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Installing huggingface-hub...")
    os.system(f"{sys.executable} -m pip install huggingface-hub -q")
    from huggingface_hub import HfApi, create_repo

HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_USERNAME: str = os.getenv("HF_USERNAME", "")
SPACE_NAME: str = "data-cleaning-env"

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is not set.")
    print("Run: export HF_TOKEN='hf_...'")
    sys.exit(1)

if not HF_USERNAME:
    print("[ERROR] HF_USERNAME environment variable is not set.")
    print("Run: export HF_USERNAME='your-huggingface-username'")
    sys.exit(1)

REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

print(f"Deploying to HuggingFace Spaces: {REPO_ID}")
print("=" * 50)

api = HfApi(token=HF_TOKEN)

# Create the Space (Docker SDK, public)
print("Step 1/2: Creating/verifying HuggingFace Space...")
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,  # does not fail if space already exists
    )
    print(f"  Space ready: https://huggingface.co/spaces/{REPO_ID}")
except Exception as exc:
    print(f"  [ERROR] Could not create space: {exc}")
    sys.exit(1)

# Upload all project files
print("Step 2/2: Uploading project files...")
ignore_patterns = [
    "*.pyc",
    "__pycache__",
    ".git",
    ".gitignore",
    "venv",
    ".venv",           # ← virtual environment — never upload
    ".venv/*",
    "outputs",
    "logs",
    "*.log",
    "deploy_to_hf.py",
    ".env",
    ".env.*",
    "secrets.*",
    "test_*.py",
    "*.egg-info",
    "*.egg-info/*",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
]

try:
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        commit_message="Deploy Data Cleaning Agent Environment",
    )
    print("  All files uploaded successfully.")
except Exception as exc:
    print(f"  [ERROR] Upload failed: {exc}")
    sys.exit(1)

print()
print("=" * 50)
print("DEPLOYMENT COMPLETE")
print("=" * 50)
space_url = f"https://huggingface.co/spaces/{REPO_ID}"
api_url = f"https://{HF_USERNAME}-{SPACE_NAME}.hf.space"
print(f"Space URL   : {space_url}")
print(f"API URL     : {api_url}")
print(f"Health check: {api_url}/health")
print(f"API docs    : {api_url}/docs")
print()
print("HuggingFace will now build the Docker image.")
print("This takes 2–5 minutes. Monitor the build log at:")
print(f"  {space_url} → click 'Logs' tab")
print()
print("Once the space shows 'Running', test it:")
print(f"  curl {api_url}/health")
print(f"  curl -X POST '{api_url}/reset?task_id=1'")