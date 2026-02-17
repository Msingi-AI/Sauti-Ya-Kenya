"""CLI to download WAXAL and save the Swahili subset to disk.

Example usage:

python examples/download_waxal.py --hf-repo msingi/waxal --out data/processed/waxal_swahili

If you don't have access to a HF repo id, point `--local` to a local WAXAL folder.
"""
import argparse
import logging
from src.sauti.data import prepare_waxal_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face dataset id for WAXAL")
    parser.add_argument("--local", type=str, default=None, help="Local path to WAXAL unpacked data")
    parser.add_argument("--out", type=str, default="data/processed/waxal_swahili", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.hf_repo and not args.local:
        print("Specify --hf-repo or --local. See README for WAXAL links.")
        return

    prepare_waxal_dataset(hf_repo=args.hf_repo, local_path=args.local, out_dir=args.out)


if __name__ == "__main__":
    main()
