"""Dataset sample definitions and chronological splitting for marine AI."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real

from .contracts import MarinePredictionRequest, MarinePredictionResult


@dataclass(frozen=True)
class MarineDatasetSample:
    """One marine AI example containing model inputs, targets, and provenance."""

    sample_id: str
    request: MarinePredictionRequest
    target: MarinePredictionResult
    source: str
    source_version: str
    license_name: str

    def __post_init__(self):
        for name in (
            "sample_id",
            "source",
            "source_version",
            "license_name",
        ):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

        if not isinstance(self.request, MarinePredictionRequest):
            raise ValueError("request must be a MarinePredictionRequest.")  # noqa: TRY004

        if not isinstance(self.target, MarinePredictionResult):
            raise ValueError("target must be a MarinePredictionResult.")  # noqa: TRY004

        self.target.validate_against(self.request)

    @property
    def forecast_start(self) -> datetime:
        """Return the timezone-aware forecast start time."""
        return datetime.fromisoformat(
            self.request.forecast.start_time.replace("Z", "+00:00")  # noqa: FURB162
        )

    @property
    def forecast_end(self) -> datetime:
        """Return the end of the target forecast period."""
        duration = timedelta(
            seconds=(self.request.forecast.steps * self.request.forecast.step_seconds)
        )
        return self.forecast_start + duration


@dataclass(frozen=True)
class MarineDatasetSplit:
    """Chronologically separated samples for model development."""

    train: tuple[MarineDatasetSample, ...]
    validation: tuple[MarineDatasetSample, ...]
    test: tuple[MarineDatasetSample, ...]

    @property
    def total_samples(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


def split_dataset(
    samples,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> MarineDatasetSplit:
    """
    Split samples chronologically and purge forecast windows crossing
    train/validation or validation/test boundaries.

    Samples excluded by the purge are intentionally not assigned to a split.
    """
    for name, ratio in (
        ("train_ratio", train_ratio),
        ("validation_ratio", validation_ratio),
    ):
        if isinstance(ratio, bool) or not isinstance(ratio, Real) or not 0 < ratio < 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("Training and validation ratios must sum to less than 1.")

    samples = list(samples)

    if len(samples) < 3:
        raise ValueError("At least three samples are required.")

    if any(not isinstance(sample, MarineDatasetSample) for sample in samples):
        raise ValueError("All samples must be MarineDatasetSample instances.")

    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Sample IDs must be unique.")

    ordered = sorted(samples, key=lambda sample: sample.forecast_start)

    # Require a consistent timezone representation for comparisons.
    timezone = ordered[0].forecast_start.utcoffset()
    if any(sample.forecast_start.utcoffset() != timezone for sample in ordered):
        raise ValueError("All samples must use the same timezone offset.")

    train_end = int(len(ordered) * train_ratio)
    validation_end = int(len(ordered) * (train_ratio + validation_ratio))

    if train_end < 1 or validation_end <= train_end:
        raise ValueError("Dataset is too small for the requested split ratios.")

    if validation_end >= len(ordered):
        raise ValueError("The requested ratios must leave samples for the test split.")

    validation_boundary = ordered[train_end].forecast_start
    test_boundary = ordered[validation_end].forecast_start

    # Exclude earlier samples whose target periods cross split boundaries.
    train = tuple(
        sample
        for sample in ordered[:train_end]
        if sample.forecast_end <= validation_boundary
    )

    validation = tuple(
        sample
        for sample in ordered[train_end:validation_end]
        if sample.forecast_end <= test_boundary
    )

    test = tuple(ordered[validation_end:])

    if not train or not validation or not test:
        raise ValueError(
            "Splitting produced an empty partition. "
            "Provide more samples or adjust the ratios."
        )

    return MarineDatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )
