from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

import numpy as np

EmbeddingsResult = NamedTuple(
    "EmbeddingsResult",
    [("embeddings", np.ndarray), ("labels", np.ndarray)]
)


@dataclass(frozen=True)
class QuantitativeMetrics:
    accuracy: float
    macro_auc: float
    micro_auc: float

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.4f}, "
            f"Macro AUC: {self.macro_auc:.4f}, "
            f"Micro AUC: {self.micro_auc:.4f}"
        )


@dataclass(frozen=True)
class ValidationResults:
    quantitative_metrics: QuantitativeMetrics
    embeddings: EmbeddingsResult = None


@dataclass(frozen=True)
class ValidationBranchTuple:
    with_object_branch: ValidationResults
    without_object_branch: ValidationResults


class DatasetValidation:
    def __init__(
        self,
        name: str,
    ):
        self._name = name
        self._validation_results: Dict[int, ValidationBranchTuple] = {}
        self._validation_partials = [100, 80, 60, 40, 20]

    def __setitem__(
        self,
        validation_partial: int,
        value: ValidationBranchTuple
    ) -> None:
        if validation_partial in self._validation_partials:
            self._validation_results[validation_partial] = value
        else:
            raise ValueError(
                f"Invalid validation partial: {validation_partial}"
            )

    def __getitem__(
        self,
        validation_partial: int
    ) -> Optional[ValidationBranchTuple]:
        if validation_partial in self._validation_partials:
            return self._validation_results.get(validation_partial)
        else:
            raise ValueError(
                f"Invalid validation partial: {validation_partial}"
            )
