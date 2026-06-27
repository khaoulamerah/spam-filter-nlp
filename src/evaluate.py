
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


# ────────────────────────────
# Style global des graphiques
# ──────────────────────────────
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({
    "figure.dpi"      : 150,      
    "figure.facecolor": "white",  
    "font.size"       : 11,
})


class ModelEvaluator:
    def __init__(self, output_dir: str = "reports"):

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)  # Crée le dossier si nécessaire

    def plot_confusion_matrix(self, result: dict, y_test, save: bool = True) -> str:
        cm = result["confusion_matrix"]
        vec_name   = result["vectorizer"]
        model_name = result["model"]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ham (0)", "Spam (1)"],
            yticklabels=["Ham (0)", "Spam (1)"],
            ax=ax,
        )

        ax.set_xlabel("Valeur prédite", fontsize=12)
        ax.set_ylabel("Valeur réelle", fontsize=12)
        ax.set_title(
            f"Matrice de confusion\n{vec_name} + {model_name}\n"
            f"F1={result['f1_score']:.4f} | Recall={result['recall']:.4f}",
            fontsize=11,
        )

        plt.tight_layout()

        filepath = ""
        if save:
            filename = f"confusion_matrix__{vec_name}__{model_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, bbox_inches="tight")
            print(f"  📊 Matrice sauvegardée : {filepath}")

        plt.close(fig)  
        return filepath

    def plot_metrics_comparison(self, results: list, save: bool = True) -> str:
        cols = ["vectorizer", "model", "accuracy", "precision", "recall", "f1_score"]
        df = pd.DataFrame(results)[cols]
        df["label"] = df["vectorizer"] + "\n" + df["model"]
        df = df.sort_values("f1_score", ascending=False)

        metrics = ["f1_score", "precision", "recall", "accuracy"]
        labels  = ["F1-Score", "Precision", "Recall", "Accuracy"]
        colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten() 

        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            ax = axes[i]

            bars = ax.barh(df["label"], df[metric], color=color, alpha=0.8, edgecolor="white")

            for bar, val in zip(bars, df[metric]):
                ax.text(
                    val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center", ha="left", fontsize=9,
                )

            ax.set_xlabel(label, fontsize=11)
            ax.set_xlim(0, 1.12)
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.5, label="Seuil 90%")

        fig.suptitle(
            "Comparaison des modèles — Phase 1 : BoW & TF-IDF",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()

        filepath = ""
        if save:
            filepath = os.path.join(self.output_dir, "metrics_comparison.png")
            fig.savefig(filepath, bbox_inches="tight")
            print(f" Comparaison sauvegardée : {filepath}")

        plt.close(fig)
        return filepath

    def plot_training_time(self, results: list, save: bool = True) -> str:
        df = pd.DataFrame(results)[["vectorizer", "model", "train_time_s", "f1_score"]]
        df["label"] = df["vectorizer"] + "\n" + df["model"]
        df = df.sort_values("train_time_s", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.RdYlGn(df["f1_score"].values)

        bars = ax.barh(df["label"], df["train_time_s"], color=colors, edgecolor="white")

        for bar, val in zip(bars, df["train_time_s"]):
            ax.text(
                val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s",
                va="center", ha="left", fontsize=10,
            )

        ax.set_xlabel("Temps d'entraînement (secondes)", fontsize=12)
        ax.set_title(
            "Temps d'entraînement par modèle\n(couleur = F1-Score : vert=élevé, rouge=faible)",
            fontsize=12,
        )
        plt.tight_layout()

        filepath = ""
        if save:
            filepath = os.path.join(self.output_dir, "training_time.png")
            fig.savefig(filepath, bbox_inches="tight")
            print(f"  Temps d'entraînement sauvegardé : {filepath}")

        plt.close(fig)
        return filepath

    def print_classification_report(self, result: dict, y_test):
        print(f"\n{'─'*50}")
        print(f"Rapport : {result['vectorizer']} + {result['model']}")
        print(f"{'─'*50}")
        print(classification_report(
            y_test,
            result["y_pred"],
            target_names=["Ham (0)", "Spam (1)"],
            digits=4,
        ))

    def generate_all_reports(self, results: list, y_test):
        print("\n" + "=" * 70)
        print("GÉNÉRATION DES RAPPORTS")
        print("=" * 70)

        # Matrice de confusion pour chaque modèle
        for result in results:
            self.plot_confusion_matrix(result, y_test)

        # Graphiques globaux
        self.plot_metrics_comparison(results)
        self.plot_training_time(results)

        # Rapport texte du meilleur modèle
        best_result = max(results, key=lambda r: r["f1_score"])
        self.print_classification_report(best_result, y_test)

        print(f"\n Tous les rapports sont dans : {self.output_dir}/")