"""Publish the Streamlit demo as a Hugging Face Space.

Run after scripts/upload_models.py, from the project root:
    python scripts/deploy_space.py

Re-run it whenever the app code changes; it uploads the latest files and the
Space rebuilds itself.
"""

from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

SPACE_REPO = "temiloluwadeboye/drug-review-sentiment"

# Only what the online app needs (no data, notebooks or models).
# Left: file in this project. Right: where it goes in the Space.
FILES = {
    "deploy/SPACE_README.md": "README.md",
    "deploy/Dockerfile": "Dockerfile",
    "deploy/requirements.txt": "deploy/requirements.txt",
    ".streamlit/config.toml": ".streamlit/config.toml",
    "src/__init__.py": "src/__init__.py",
    "src/lstm.py": "src/lstm.py",
    "src/predict.py": "src/predict.py",
    "app/__init__.py": "app/__init__.py",
    "app/streamlit_app.py": "app/streamlit_app.py",
}


def main():
    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)

    # Upload every file in one commit, so the Space rebuilds once
    operations = [
        CommitOperationAdd(path_in_repo=target, path_or_fileobj=Path(source))
        for source, target in FILES.items()
    ]
    api.create_commit(
        repo_id=SPACE_REPO,
        repo_type="space",
        operations=operations,
        commit_message="Deploy Streamlit demo",
    )
    print(f"Uploaded. The Space is now building (about 5-10 minutes):")
    print(f"https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
