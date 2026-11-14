"""Train and evaluate an offshore regional FNO candidate."""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from port_twin_app.services.marine_ai.evaluation import evaluate_checkpoint
from port_twin_app.services.marine_ai.model import (
    PREDICTION_MODE_RESIDUAL,
    MarineFNO,
)
from port_twin_app.services.marine_ai.serialization import (
    load_prepared_dataset,
)
from port_twin_app.services.marine_ai.training import (
    TrainingConfig,
    train_model,
)

DATASET_PATH = PROJECT_ROOT / "data/processed/marine/offshore_training_dataset.npz"

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts/marine_fno_offshore_candidate.pt"

REPORT_PATH = PROJECT_ROOT / "artifacts/offshore_fno_evaluation.json"

SEED = 42
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cpu"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    # Seed before model initialization for reproducibility.
    torch.manual_seed(SEED)

    if torch.cuda.is_available() and DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("Loading offshore training dataset...")
    dataset = load_prepared_dataset(DATASET_PATH)

    print(f"Training samples: {len(dataset.split.train)}")
    print(f"Validation samples: {len(dataset.split.validation)}")
    print(f"Test samples: {len(dataset.split.test)}")

    print("\nInitializing FNO...")
    model = MarineFNO(
        width=24,
        modes_x=8,
        modes_y=8,
        layers=4,
    )

    config = TrainingConfig(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        checkpoint_path=str(CHECKPOINT_PATH),
        seed=SEED,
        prediction_mode=PREDICTION_MODE_RESIDUAL,
    )

    print("\nStarting training...")
    training_result = train_model(
        model,
        dataset.split,
        config,
    )

    print("\nTraining completed.")
    print(f"Model version: {training_result.model_version}")
    print(f"Best epoch: {training_result.best_epoch}")
    print(f"Best validation loss: {training_result.best_validation_loss:.6f}")
    print(f"Checkpoint: {training_result.checkpoint_path}")

    # Attach dataset identity to this candidate checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint["training_data"] = {
        "dataset": "offshore_training_dataset",
        "domain": "Mombasa-adjacent offshore regional domain",
        "grid_nx": dataset.metadata["grid"]["nx"],
        "grid_ny": dataset.metadata["grid"]["ny"],
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "seed": SEED,
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    print("\nEvaluating against the held-out test partition...")
    evaluation = evaluate_checkpoint(
        CHECKPOINT_PATH,
        dataset.split.test,
        device=DEVICE,
    )

    print(f"Samples evaluated: {evaluation.samples_evaluated}")

    report = {
        "model_version": evaluation.model_version,
        "candidate_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "domain": "Mombasa-adjacent offshore regional domain",
        "grid": dataset.metadata["grid"],
        "training": {
            "epochs": training_result.epochs_completed,
            "best_epoch": training_result.best_epoch,
            "best_validation_loss": (training_result.best_validation_loss),
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "prediction_mode": PREDICTION_MODE_RESIDUAL,
        },
        "evaluation": {
            "samples": evaluation.samples_evaluated,
            "note": evaluation.evaluation_note,
            "fno": {
                name: {
                    "mae": metrics.mae,
                    "rmse": metrics.rmse,
                }
                for name, metrics in evaluation.fno_metrics.items()
            },
            "persistence": {
                name: {
                    "mae": metrics.mae,
                    "rmse": metrics.rmse,
                }
                for name, metrics in evaluation.baseline_metrics.items()
            },
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nTest evaluation results:")
    print("-" * 65)

    for field in evaluation.fno_metrics:
        fno = evaluation.fno_metrics[field]
        baseline = evaluation.baseline_metrics[field]

        print(f"\n{field}")
        print(f"  FNO        MAE: {fno.mae:.6f} | RMSE: {fno.rmse:.6f}")
        print(f"  Persistence MAE: {baseline.mae:.6f} | RMSE: {baseline.rmse:.6f}")

        print(
            "  FNO better on both metrics: "
            f"{fno.mae < baseline.mae and fno.rmse < baseline.rmse}"
        )

    print("\nTraining and evaluation finished.")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Evaluation report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
