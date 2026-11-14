import sys

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parents[1])

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import torch

from port_twin_app.services.marine_ai.model import (
    load_checkpoint,
    predict,
)
from port_twin_app.services.marine_ai.serialization import (
    load_prepared_dataset,
)

DATASET_PATH = Path("data/processed/marine/offshore_training_dataset.npz")
CHECKPOINT_PATH = Path("artifacts/marine_fno_offshore_candidate.pt")

FIELDS = (
    ("water_level_m", "m"),
    ("current_u_ms", "m/s"),
    ("current_v_ms", "m/s"),
)


def result_tensor(result):
    """Convert prediction fields to [steps, 3, ny, nx]."""
    return torch.tensor(
        [
            result.water_level_m,
            result.current_u_ms,
            result.current_v_ms,
        ],
        dtype=torch.float32,
    ).permute(1, 0, 2, 3)


def main():
    print("Loading offshore dataset...")
    dataset = load_prepared_dataset(DATASET_PATH)
    samples = dataset.split.test

    print(f"Test samples: {len(samples)}")

    print("\nLoading candidate checkpoint...")
    model, checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        device="cpu",
    )

    print(f"Model version: {checkpoint['model_version']}")
    print(f"Prediction mode: {checkpoint.get('prediction_mode', 'absolute')}")

    predictions = []
    baselines = []
    targets = []

    print("\nRunning inference on held-out test samples...")

    for sample in samples:
        request = sample.request

        prediction = predict(
            request,
            model,
            device="cpu",
            target_mean=checkpoint.get("target_mean"),
            target_std=checkpoint.get("target_std"),
            input_mean=checkpoint.get("input_mean"),
            input_std=checkpoint.get("input_std"),
            prediction_mode=checkpoint.get("prediction_mode", "absolute"),
        )
        prediction.validate_against(request)
        sample.target.validate_against(request)

        predictions.append(result_tensor(prediction))
        targets.append(result_tensor(sample.target))

        initial = torch.stack(
            [
                torch.tensor(request.initial_water_level_m),
                torch.tensor(request.initial_current_u_ms),
                torch.tensor(request.initial_current_v_ms),
            ]
        ).float()

        baselines.append(
            initial.unsqueeze(0).expand(request.forecast.steps, -1, -1, -1)
        )

    predictions = torch.stack(predictions)
    baselines = torch.stack(baselines)
    targets = torch.stack(targets)

    print(f"Evaluated samples: {len(samples)}")
    print("\nLEAD-TIME METRICS")
    print("Errors are aggregated across samples and grid cells.")
    print("Lower MAE/RMSE is better.\n")

    for channel, (field, unit) in enumerate(FIELDS):
        print(f"\n{field} ({unit})")
        print(
            f"{'Lead':<8}"
            f"{'FNO MAE':>14}"
            f"{'Base MAE':>14}"
            f"{'FNO RMSE':>14}"
            f"{'Base RMSE':>14}"
            f"{'MAE gain':>12}"
        )

        for lead in range(targets.shape[1]):
            fno_error = predictions[:, lead, channel] - targets[:, lead, channel]
            base_error = baselines[:, lead, channel] - targets[:, lead, channel]

            fno_mae = fno_error.abs().mean().item()
            base_mae = base_error.abs().mean().item()

            fno_rmse = fno_error.square().mean().sqrt().item()
            base_rmse = base_error.square().mean().sqrt().item()

            gain = (base_mae - fno_mae) / base_mae * 100 if base_mae > 0 else 0.0

            print(
                f"{lead + 1}h"
                f"{fno_mae:>14.6f}"
                f"{base_mae:>14.6f}"
                f"{fno_rmse:>14.6f}"
                f"{base_rmse:>14.6f}"
                f"{gain:>11.2f}%"
            )


if __name__ == "__main__":
    main()
