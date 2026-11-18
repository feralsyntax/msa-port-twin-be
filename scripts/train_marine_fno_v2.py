"""Train the forcing-conditioned marine FNO v2 model."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from port_twin_app.services.marine_ai.model_v2 import MarineFNOV2
from port_twin_app.services.marine_ai.preparation_v2 import (
    prepare_marine_fno_v2_dataset,
)
from port_twin_app.services.marine_ai.training_v2 import (
    TrainingV2Config,
    train_model_v2,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the forcing-conditioned marine FNO v2 model."
    )

    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=Path("data/processed/marine/shallow_water"),
        help="Directory containing the generated v2 NetCDF files.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/marine_fno_v2.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--continuity-residual-scale-ms",
        type=float,
        default=1e-4,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Preparing marine FNO v2 dataset...")

    dataset = prepare_marine_fno_v2_dataset(
        dataset_directory=args.dataset_directory,
        seed=args.seed,
    )

    print(f"Total samples:      {dataset.total_samples}")
    print(f"Training samples:   {len(dataset.train)}")
    print(f"Validation samples: {len(dataset.validation)}")
    print(f"Test samples:       {len(dataset.test)}")
    print("Test samples will not be used during training.")

    print("\nScenario partitions:")
    for partition in ("train", "validation", "test"):
        scenarios = sorted(
            name
            for name, assigned in dataset.scenario_partitions.items()
            if assigned == partition
        )
        print(f"  {partition}: {len(scenarios)} scenarios")

    # Seed before model construction so initial weights are reproducible.
    torch.manual_seed(args.seed)
    model = MarineFNOV2()

    config = TrainingV2Config(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        physics_loss_weight=args.physics_loss_weight,
        continuity_residual_scale_ms=args.continuity_residual_scale_ms,
    )

    print("\nStarting FNO v2 training...")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Physics loss weight: {config.physics_loss_weight}")
    print(f"Checkpoint: {config.checkpoint_path}")

    result = train_model_v2(
        model=model,
        dataset=dataset,
        config=config,
    )

    print("\nTraining completed.")
    print(f"Model version: {result.model_version}")
    print(f"Epochs completed: {result.epochs_completed}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best validation loss: {result.best_validation_loss:.8f}")

    print("\nEpoch loss history:")
    print(
        "Epoch | Train total | Validation total | "
        "Train supervised | Train continuity | "
        "Validation supervised | Validation continuity"
    )

    for index in range(result.epochs_completed):
        print(
            f"{index + 1:5d} | "
            f"{result.training_losses[index]:11.6f} | "
            f"{result.validation_losses[index]:16.6f} | "
            f"{result.training_supervised_losses[index]:16.6f} | "
            f"{result.training_continuity_losses[index]:16.6f} | "
            f"{result.validation_supervised_losses[index]:22.6f} | "
            f"{result.validation_continuity_losses[index]:22.6f}"
        )

    print(f"\nCheckpoint saved and load-verified: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
