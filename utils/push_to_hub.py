import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Push a checkpoint to a Hugging Face Hub model repo")
    parser.add_argument("checkpoint_path", help="path to the .pth checkpoint file to upload")
    parser.add_argument("repo_id", help="target Hugging Face repo, e.g. 'username/model-name'")
    parser.add_argument("--private", action="store_true", help="create the repo as private if it doesn't exist yet")
    args = parser.parse_args()

    assert os.path.exists(args.checkpoint_path), f"Checkpoint not found: {args.checkpoint_path}"

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    filename = "chatling.pth"
    api.upload_file(
        path_or_fileobj=args.checkpoint_path,
        path_in_repo=filename,
        repo_id=args.repo_id,
    )
    print(f"Uploaded {args.checkpoint_path} to https://huggingface.co/{args.repo_id}/blob/main/{filename}")


if __name__ == "__main__":
    main()
