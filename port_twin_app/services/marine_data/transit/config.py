import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DraftClass:
    """Experimental representative vessel draft and minimum UKC requirement."""

    name: str
    representative_draft_m: float
    minimum_ukc_m: float

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Draft class name cannot be empty.")

        if (
            not math.isfinite(self.representative_draft_m)
            or self.representative_draft_m <= 0
        ):
            raise ValueError("Representative draft must be finite and positive.")

        if not math.isfinite(self.minimum_ukc_m) or self.minimum_ukc_m <= 0:
            raise ValueError("Minimum UKC must be finite and positive.")


@dataclass(frozen=True)
class TransitAssessmentConfig:
    """Configurable experimental limits for transit-window assessment."""

    draft_classes: tuple[DraftClass, ...] = (
        DraftClass("small", 5.0, 1.0),
        DraftClass("medium", 10.0, 1.5),
        DraftClass("deep_draft", 15.0, 2.0),
    )
    maximum_current_speed_ms: float = 0.75
    maximum_wind_speed_ms: float = 10.0

    def __post_init__(self):
        if not self.draft_classes:
            raise ValueError("At least one draft class is required.")

        names = [draft_class.name for draft_class in self.draft_classes]

        if len(names) != len(set(names)):
            raise ValueError("Draft class names must be unique.")

        if (
            not math.isfinite(self.maximum_current_speed_ms)
            or self.maximum_current_speed_ms < 0
        ):
            raise ValueError("Maximum current speed must be finite and non-negative.")

        if (
            not math.isfinite(self.maximum_wind_speed_ms)
            or self.maximum_wind_speed_ms < 0
        ):
            raise ValueError("Maximum wind speed must be finite and non-negative.")

    def to_dict(self) -> dict:
        """Return a JSON-serializable configuration snapshot."""
        return asdict(self)
