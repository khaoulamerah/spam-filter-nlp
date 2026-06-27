
FROM python:3.11-slim-bullseye


LABEL maintainer="ton-email@exemple.com"
LABEL description="Spam Filter NLP - Phase 1 : Modèles classiques"
LABEL version="1.0"


# ÉTAPE 3 : Variables d'environnement système
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ÉTAPE 4 : Répertoire de travail dans le conteneur
WORKDIR /app

# ÉTAPE 5 : Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ÉTAPE 6 : Copier et installer les dépendances Python
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ÉTAPE 7 : Télécharger les ressources NLTK nécessaires
RUN python -c "import nltk; \
    nltk.download('stopwords', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('wordnet', quiet=True)"


# ÉTAPE 8 : Copier le code source dans le conteneur
COPY src/ ./src/
COPY data/ ./data/
COPY .env .

# ÉTAPE 9 : Créer le dossier de sauvegarde des modèles
RUN mkdir -p models


# ÉTAPE 10 : Commande par défaut
# Cette commande s'exécute quand on lance "docker run <image>".
# Elle lance le pipeline NLP complet.
CMD ["python", "src/main.py"]