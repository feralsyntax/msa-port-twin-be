import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from port_twin_app.services.marine_ai.model import load_checkpoint, predict
from port_twin_app.services.marine_ai.serialization import load_prepared_dataset

DATASET_PATH = Path("data/processed/marine/offshore_training_dataset.npz")
CHECKPOINT_PATH = Path("artifacts/marine_fno_offshore_candidate.pt")

FIELDS = (
    ("water_level_m", "m"),
    ("current_u_ms", "m/s"),
    ("current_v_ms", "m/s"),
)


def result_tensor(result):
    return torch.tensor(
        [
            result.water_level_m,
            result.current_u_ms,
            result.current_v_ms,
        ],
        dtype=torch.float32,
    ).permute(1, 0, 2, 3)


def correlation(actual, predicted):
    actual = actual.flatten()
    predicted = predicted.flatten()

    actual_centered = actual - actual.mean()
    predicted_centered = predicted - predicted.mean()

    denominator = (
        actual_centered.square().sum() * predicted_centered.square().sum()
    ).sqrt()

    if denominator == 0:
        return float("nan")

    return ((actual_centered * predicted_centered).sum() / denominator).item()


def main():
    print("Loading dataset...")
    dataset = load_prepared_dataset(DATASET_PATH)
    samples = dataset.split.test

    print(f"Held-out test samples: {len(samples)}")

    print("\nLoading candidate checkpoint...")
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH, device="cpu")

    predictions = []
    actual_changes = []

    print("Running inference...")

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

        initial = torch.stack(
            [
                torch.as_tensor(request.initial_water_level_m),
                torch.as_tensor(request.initial_current_u_ms),
                torch.as_tensor(request.initial_current_v_ms),
            ]
        ).float()

        predicted = result_tensor(prediction)
        target = result_tensor(sample.target)

        predictions.append(predicted - initial.unsqueeze(0))
        actual_changes.append(target - initial.unsqueeze(0))

    predictions = torch.stack(predictions)
    actual_changes = torch.stack(actual_changes)

    print("\nPREDICTED CHANGE VS ACTUAL CHANGE")
    print("Bias = predicted change minus actual change.")
    print("Magnitude ratio = predicted mean absolute change / actual.")
    print("Correlation is calculated across test samples and grid cells.")

    for channel, (field, unit) in enumerate(FIELDS):
        print(f"\n{field} ({unit})")
        print(
            f"{'Lead':<6}"
            f"{'Actual |Δ|':>13}"
            f"{'Predicted |Δ|':>16}"
            f"{'Bias':>13}"
            f"{'Magnitude':>13}"
            f"{'Corr.':>10}"
        )

        for lead in range(actual_changes.shape[1]):
            actual = actual_changes[:, lead, channel]
            predicted = predictions[:, lead, channel]

            actual_magnitude = actual.abs().mean().item()
            predicted_magnitude = predicted.abs().mean().item()
            bias = (predicted - actual).mean().item()

            magnitude_ratio = (
                predicted_magnitude / actual_magnitude
                if actual_magnitude > 0
                else float("nan")
            )

            corr = correlation(actual, predicted)

            print(
                f"{lead + 1}h"
                f"{actual_magnitude:>13.6f}"
                f"{predicted_magnitude:>16.6f}"
                f"{bias:>13.6f}"
                f"{magnitude_ratio:>12.2f}x"
                f"{corr:>10.3f}"
            )


if __name__ == "__main__":
    main()
