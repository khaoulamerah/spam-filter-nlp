

import os
import sys
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessor import TextPreprocessor
from models import NLPTrainer
from evaluate import ModelEvaluator



DATA_PATH    = os.getenv("DATA_PATH",    "data/sms_spam.csv")
MODELS_DIR   = os.getenv("MODELS_DIR",   "models")
REPORTS_DIR  = os.getenv("REPORTS_DIR",  "reports")
LANGUAGE     = os.getenv("LANGUAGE",     "english")
TEST_SIZE    = float(os.getenv("TEST_SIZE",    "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE",   "42"))


# ──────────────────────────────────
# ÉTAPE 1 : Chargement des données
# ──────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:

    print(f"\n{'='*60}")
    print("ÉTAPE 1 : CHARGEMENT DES DONNÉES")
    print(f"{'='*60}")
    print(f"📂 Fichier : {filepath}")

    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="latin-1",
        )
    except FileNotFoundError:
        print(f"\n❌ ERREUR : Fichier introuvable → {filepath}")
        print("   Télécharge le dataset depuis :")
        print("   https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
        print("   et place-le dans data/sms_spam.csv")
        sys.exit(1)

 
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Suppression des lignes avec des valeurs manquantes
    df = df.dropna().reset_index(drop=True)

    print(f"✅ {len(df)} messages chargés")
    return df


# ───────────────────────────────────────────
# ÉTAPE 2 : Exploration rapide (EDA console)
# ───────────────────────────────────────────
def quick_eda(df: pd.DataFrame):

    print(f"\n{'='*60}")
    print("ÉTAPE 2 : EXPLORATION DES DONNÉES (EDA)")
    print(f"{'='*60}")

    # Distribution des classes
    counts = df["label"].value_counts()
    ham_count  = counts.get(0, 0)
    spam_count = counts.get(1, 0)
    total = len(df)

    print(f"\n📊 Distribution des classes :")
    print(f"   Ham  (0) : {ham_count:>5} messages ({ham_count/total*100:.1f}%)")
    print(f"   Spam (1) : {spam_count:>5} messages ({spam_count/total*100:.1f}%)")
    print(f"   Total    : {total:>5} messages")


    ratio = ham_count / spam_count
    print(f"\n⚠️  Ratio Ham/Spam : {ratio:.1f}:1")
    if ratio > 3:
        print("   Dataset déséquilibré → privilégier F1-Score sur Accuracy")

    # Longueur des messages
    df["text_length"] = df["text"].str.len()
    print(f"\n📏 Longueur des messages (caractères) :")
    print(f"   Ham  — moyenne : {df[df['label']==0]['text_length'].mean():.0f} | "
          f"max : {df[df['label']==0]['text_length'].max()}")
    print(f"   Spam — moyenne : {df[df['label']==1]['text_length'].mean():.0f} | "
          f"max : {df[df['label']==1]['text_length'].max()}")

    # Exemples
    print(f"\n📧 Exemple de Ham :")
    print(f"   \"{df[df['label']==0]['text'].iloc[0][:100]}...\"")
    print(f"\n🚨 Exemple de Spam :")
    print(f"   \"{df[df['label']==1]['text'].iloc[0][:100]}...\"")

    # Nettoyage de la colonne temporaire
    df.drop(columns=["text_length"], inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 : Prétraitement du texte
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, language: str) -> pd.DataFrame:

    print(f"\n{'='*60}")
    print("ÉTAPE 3 : PRÉTRAITEMENT DU TEXTE")
    print(f"{'='*60}")

    preprocessor = TextPreprocessor(language=language)

    print("🔄 Nettoyage en cours...", end=" ", flush=True)
    df["text_clean"] = preprocessor.clean_series(df["text"])
    print("✅")

    # Vérification : affiche un exemple avant/après
    idx = df[df["label"] == 1].index[0]  # Premier spam
    print(f"\n📝 Exemple de transformation (Spam) :")
    print(f"   Avant  : \"{df.loc[idx, 'text'][:80]}...\"")
    print(f"   Après  : \"{df.loc[idx, 'text_clean'][:80]}\"")

    # Suppression des textes vides après nettoyage (cas rares)
    empty_count = (df["text_clean"].str.strip() == "").sum()
    if empty_count > 0:
        print(f"\n⚠️  {empty_count} texte(s) vide(s) après nettoyage → supprimés")
        df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)

    print(f"\n✅ {len(df)} textes nettoyés")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 : Entraînement et évaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("ÉTAPE 4 : ENTRAÎNEMENT DES MODÈLES")
    print(f"{'='*60}")

    # ── Initialisation du trainer ─────────────────────────────────────────────
    trainer = NLPTrainer(
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
    )

    # ── Division train/test ───────────────────────────────────────────────────
    X = df["text_clean"]   # Features : textes nettoyés
    y = df["label"]        # Target   : 0=ham, 1=spam

    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    print(f"\n📊 Split train/test :")
    print(f"   Train : {len(X_train)} exemples")
    print(f"   Test  : {len(X_test)} exemples")
    print(f"   Spam dans le test : {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    # ── Entraînement de toutes les combinaisons ───────────────────────────────
    # 2 vectoriseurs × 4 modèles = 8 pipelines
    results = trainer.train_all(X_train, X_test, y_train, y_test)

    # ── Tableau comparatif ───
    trainer.print_comparison()

    # ── Génération des rapports et graphiques ──
    print(f"\n{'='*60}")
    print("ÉTAPE 5 : GÉNÉRATION DES RAPPORTS")
    print(f"{'='*60}")

    evaluator = ModelEvaluator(output_dir=REPORTS_DIR)
    evaluator.generate_all_reports(results, y_test)

    return trainer, evaluator, y_test


# ──────────────────────────────────────
# ÉTAPE 5 : Sauvegarde du meilleur modèle
# ──────────────────────────────────────

def save_best(trainer: NLPTrainer):

    print(f"\n{'='*60}")
    print("ÉTAPE 6 : SAUVEGARDE DU MEILLEUR MODÈLE")
    print(f"{'='*60}")

    filepath = trainer.save_best_model(models_dir=MODELS_DIR)
    print(f" Modèle prêt à être rechargé depuis : {filepath}")


# ──────────────────────────────────
# Point d'entrée principal
# ──────────────────────────────────

def main():

    print("\n" + "🚀 " * 20)
    print("SPAM FILTER NLP — PHASE 1 : MODÈLES CLASSIQUES")
    print("🚀 " * 20)

    # Exécution des étapes dans l'ordre
    df                        = load_data(DATA_PATH)
    quick_eda(df)
    df                        = preprocess(df, LANGUAGE)
    trainer, evaluator, y_test = train_and_evaluate(df)
    save_best(trainer)

    print("\n" + "✅ " * 20)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("✅ " * 20)
    print(f"\n Modèles sauvegardés dans  : {MODELS_DIR}/")
    print(f" Rapports sauvegardés dans : {REPORTS_DIR}/")
    print()


if __name__ == "__main__":
    main()