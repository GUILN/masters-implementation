from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(
    with_object_cm: np.ndarray,
    class_names: List[str],
    without_object_cm: np.ndarray = None,
    normalize=True,
    figsize=(10, 8),
    cmap="Blues"
):
    """
    Visual, attractive confusion matrix with optional normalization.
    When without_object_cm is provided, plots both matrices side by side.
    """
    if normalize:
        with_object_cm = with_object_cm.astype("float") / with_object_cm.sum(axis=1, keepdims=True)
        with_object_cm = np.nan_to_num(with_object_cm)  # handle div-by-zero
        if without_object_cm is not None:
            without_object_cm = without_object_cm.astype("float") / without_object_cm.sum(axis=1, keepdims=True)
            without_object_cm = np.nan_to_num(without_object_cm)  # handle div-by-zero

    if without_object_cm is not None:
        # Plot side by side when both matrices are provided
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.8, figsize[1]))
        
        # Plot with object confusion matrix
        sns.set(font_scale=1.0)
        sns.heatmap(
            with_object_cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            square=True,
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 10, "weight": "bold"},
            ax=ax1
        )
        
        ax1.set_title("Com Ramificação de Objetos", fontsize=16, weight="bold")
        ax1.set_ylabel("Rótulo Verdadeiro", fontsize=12)
        ax1.set_xlabel("Rótulo Previsto", fontsize=12)
        ax1.set_xticks(np.arange(len(class_names)) + 0.5)
        ax1.set_xticklabels(class_names, rotation=45, ha="right")
        ax1.set_yticks(np.arange(len(class_names)) + 0.5)
        ax1.set_yticklabels(class_names, rotation=0)
        
        # Plot without object confusion matrix
        sns.heatmap(
            without_object_cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            square=True,
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 10, "weight": "bold"},
            ax=ax2
        )
        
        ax2.set_title("Sem Ramificação de Objetos", fontsize=16, weight="bold")
        ax2.set_ylabel("Rótulo Verdadeiro", fontsize=12)
        ax2.set_xlabel("Rótulo Previsto", fontsize=12)
        ax2.set_xticks(np.arange(len(class_names)) + 0.5)
        ax2.set_xticklabels(class_names, rotation=45, ha="right")
        ax2.set_yticks(np.arange(len(class_names)) + 0.5)
        ax2.set_yticklabels(class_names, rotation=0)
        
        plt.tight_layout()
        plt.show()
    else:
        # Single confusion matrix when without_object_cm is not provided
        plt.figure(figsize=figsize)
        sns.set(font_scale=1.2)
        sns.heatmap(
            with_object_cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            square=True,
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 12, "weight": "bold"}
        )

        plt.title("Matriz de Confusão", fontsize=18, weight="bold")
        plt.ylabel("Rótulo Verdadeiro", fontsize=14)
        plt.xlabel("Rótulo Previsto", fontsize=14)

        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)

        plt.tight_layout()
        plt.show()
