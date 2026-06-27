
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


class TextPreprocessor:

    def __init__(self, language: str = "english"):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)

    # ------------------------------------------------------------------
    # Etapes de nettoyage (methodes privees, prefixe _)
    # ------------------------------------------------------------------

    def _to_lowercase(self, text: str) -> str:
        return text.lower()

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\S+", "", text)

    def _remove_special_characters(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z\s]", "", text)

    def _tokenize(self, text: str) -> list:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: list) -> list:
        return [
            token for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]

    def _lemmatize(self, tokens: list) -> list:

        return [self.lemmatizer.lemmatize(token) for token in tokens]


    def clean(self, text: str) -> str:
        """
        Pipeline complet de nettoyage d'un texte.

        Ordre des etapes :
        1. Minuscules
        2. Suppression URLs
        3. Caracteres speciaux
        4. Tokenisation
        5. Stopwords
        6. Lemmatisation
        7. Reconstruction en chaine

        """
        if not isinstance(text, str):
            return ""

        text = self._to_lowercase(text)
        text = self._remove_urls(text)
        text = self._remove_special_characters(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)

        return " ".join(tokens)

    def clean_series(self, series):

        return series.apply(self.clean)