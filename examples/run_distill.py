"""Example CLI to demonstrate the distillation scaffold."""
import argparse
import logging
from src.sauti.data import prepare_waxal_dataset
from src.sauti.distill import Distiller


def main():
    parser = argparse.ArgumentParser(description="Run Sauti distillation scaffold")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face WAXAL repo id")
    parser.add_argument("--local", type=str, default=None, help="Local WAXAL path")
    parser.add_argument("--teacher", type=str, default="facebook/fish-speech-1.5", help="Teacher model id")
    parser.add_argument("--student", type=str, default="cosy/cosyvoice-0.5b", help="Student model id")
    parser.add_argument("--epochs", type=int, default=1, help="Distillation epochs (demo)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data_path = prepare_waxal_dataset(hf_repo=args.hf_repo, local_path=args.local)

    # Placeholder: train_dataset should be an iterator of tokenized batches
    train_dataset = []

    distiller = Distiller(teacher_name=args.teacher, student_name=args.student)
    distiller.distill(train_dataset, epochs=args.epochs, output_dir="models/sauti_student")


if __name__ == "__main__":
    main()
