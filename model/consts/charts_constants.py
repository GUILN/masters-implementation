# Object Branch
from dataclasses import dataclass


@dataclass(frozen=True)
class ChartsLabels:
    with_object_branch: str
    without_object_branch: str
    confusion_matrix_true: str
    confusion_matrix_predicted: str
    quantitative_observation_rate: str
    quantitative_auc: str
    quantitative_auc_progression_over_observation_rate: str
    quantitative_accuracy: str
    quantitative_accuracy_progression_over_observation_rate: str


portuguese_labels = ChartsLabels(
    with_object_branch="Com Ramificação de Objetos",
    without_object_branch="Sem Ramificação de Objetos",
    confusion_matrix_true="Classe Verdadeira",
    confusion_matrix_predicted="Classe Predita",
    quantitative_observation_rate="Taxa de Observação (%)",
    quantitative_auc="AUC",
    quantitative_auc_progression_over_observation_rate="Progressão do AUC com a Taxa de Observação",
    quantitative_accuracy="Acurácia",
    quantitative_accuracy_progression_over_observation_rate="Progressão da Acurácia com a Taxa de Observação",
)


CHARTS_LABELS: ChartsLabels = portuguese_labels
