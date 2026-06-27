
import os
import time                      
import joblib      # Sauvegarde et chargement des modèles

import numpy as np
import pandas as pd

# ─── Vectorisation 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# ─── Modèles 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# ─── Pipeline & Évaluation 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ────────────────────────────────────
# Configuration des vectoriseurs
# ────────────────────────────────────

def get_vectorizers() -> dict:
    return {
        "bag_of_words": CountVectorizer(
            max_features=5000,   
            ngram_range=(1, 1), 
            min_df=2,           
        ),
        "tfidf": TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 1),
            min_df=2,
            sublinear_tf=True,   
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Configuration des modèles
# ─────────────────────────────────────────────────────────────────────────────

def get_models(random_state: int = 42) -> dict:
    return {
        "naive_bayes": MultinomialNB(
            alpha=1.0 
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,         
            random_state=random_state,
            C=1.0,             
        ),
        "svm": LinearSVC(
            max_iter=2000,
            random_state=random_state,
            C=1.0,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,    
            random_state=random_state,
            n_jobs=-1,          
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale : NLPTrainer
# ─────────────────────────────────────────────────────────────────────────────

class NLPTrainer:

    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.vectorizers = get_vectorizers()
        self.models = get_models(random_state)
        self.results = []      
        self.best_pipeline = None  
        self.best_f1 = 0.0

    def split_data(self, X, y):
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,       
        )

    def train_single(
        self,
        vec_name: str,
        vectorizer,
        model_name: str,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    ) -> dict:
        
        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", model),
        ])

        # ── Entraînement avec mesure du temps
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        # ── Prédictions 
        y_pred = pipeline.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        # ── Validation croisée (fiabilité du score) ───────────────────────────
        # StratifiedKFold garantit que chaque fold conserve la proportion
        # spam/ham du dataset. On adapte n_folds au nombre d'exemples.
        from sklearn.model_selection import StratifiedKFold
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])
        min_class_count = int(y_all.value_counts().min())
        n_folds = min(5, min_class_count)

        pipeline_cv = Pipeline([
            ("vectorizer", vectorizer.__class__(**vectorizer.get_params())),
            ("classifier", model.__class__(**model.get_params())),
        ])

        if n_folds >= 2:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(pipeline_cv, X_all, y_all, cv=cv, scoring="f1")
        else:
            cv_scores = np.array([f1])

        # ── Sauvegarde du meilleur pipeline ──────────────────────────────────
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_pipeline = pipeline
            self.best_pipeline_name = f"{vec_name}__{model_name}"

        result = {
            "vectorizer"   : vec_name,
            "model"        : model_name,
            "accuracy"     : round(accuracy, 4),
            "precision"    : round(precision, 4),
            "recall"       : round(recall, 4),
            "f1_score"     : round(f1, 4),
            "cv_f1_mean"   : round(cv_scores.mean(), 4),
            "cv_f1_std"    : round(cv_scores.std(), 4),
            "train_time_s" : round(train_time, 3),
            "pipeline"     : pipeline,
            "y_pred"       : y_pred,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        self.results.append(result)
        return result

    def train_all(self, X_train, X_test, y_train, y_test) -> list:

        print("=" * 70)
        print("ENTRAÎNEMENT DES MODÈLES")
        print("=" * 70)

        for vec_name, vectorizer in self.vectorizers.items():
            print(f"\n📐 Vectoriseur : {vec_name.upper()}")
            print("-" * 50)

            for model_name, model in self.models.items():
                print(f"  🔄 Entraînement : {model_name}...", end=" ", flush=True)

                result = self.train_single(
                    vec_name, vectorizer,
                    model_name, model,
                    X_train, X_test, y_train, y_test,
                )

                print(
                    f"F1={result['f1_score']:.4f} | "
                    f"Recall={result['recall']:.4f} | "
                    f"Temps={result['train_time_s']:.3f}s"
                )

        print(f"\n🏆 Meilleur modèle : {self.best_pipeline_name} (F1={self.best_f1:.4f})")
        return self.results

    def get_comparison_dataframe(self) -> pd.DataFrame:
        cols = [
            "vectorizer", "model", "accuracy", "precision",
            "recall", "f1_score", "cv_f1_mean", "cv_f1_std", "train_time_s"
        ]
        df = pd.DataFrame(self.results)[cols]
        return df.sort_values("f1_score", ascending=False).reset_index(drop=True)

    def print_comparison(self):
        """Affiche le tableau comparatif dans la console."""
        df = self.get_comparison_dataframe()
        print("\n" + "=" * 70)
        print("TABLEAU COMPARATIF DES MODÈLES")
        print("=" * 70)
        print(df.to_string(index=False))
        print()

    def save_best_model(self, models_dir: str = "models") -> str:
        os.makedirs(models_dir, exist_ok=True)  

        filename = f"best_model__{self.best_pipeline_name}.joblib"
        filepath = os.path.join(models_dir, filename)

        # joblib.dump() sérialise l'objet Python en fichier binaire
        joblib.dump(self.best_pipeline, filepath)

        print(f" Modèle sauvegardé : {filepath}")
        return filepath