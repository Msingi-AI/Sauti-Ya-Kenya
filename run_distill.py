import logging

from src.sauti.distill_pipeline import run_distillation

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    run_distillation("configs/distill.yaml")
