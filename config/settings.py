"""Configuration centralisée du module Agent IA (Membre 3).

Toutes les valeurs viennent des variables d'environnement (fichier .env à la
racine du projet, voir .env.example). Aucune clé API ni identifiant n'est
codé en dur ici, conformément au cahier des charges.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get_bool(nom: str, defaut: bool) -> bool:
    valeur = os.getenv(nom)
    if valeur is None:
        return defaut
    return valeur.strip().lower() in {"1", "true", "yes", "oui"}


def _get_int(nom: str, defaut: int) -> int:
    valeur = os.getenv(nom)
    if valeur is None or not valeur.strip():
        return defaut
    try:
        return int(valeur)
    except ValueError:
        return defaut


# --- Backend de recherche -----------------------------------------------
# "mock"  : corpus simulé, aucune dépendance Oracle (valeur par défaut).
# "real"  : délègue au module search/ du Membre 2 (nécessite Oracle + .env
#           avec ORACLE_USER / ORACLE_PWD / ORACLE_DSN).
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "mock")

# --- Fournisseur LLM -------------------------------------------------------
# "groq"   : gratuit (free tier généreux), modèles Llama avec appel d'outils
#            — recommandé par défaut pour ce projet étudiant.
# "openai" : nécessite une clé payante, utilisé si LLM_PROVIDER=openai.
# "fake"   : aucun appel réseau, répond avec un texte fixe — utilisé par les
#            tests automatiques et les démonstrations hors-ligne.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LLM_TEMPERATURE = _get_int("LLM_TEMPERATURE", 0)  # entier car .env; 0 = déterministe

# --- Paramètres de recherche / agent ---------------------------------------
DEFAULT_TOP_K = _get_int("DEFAULT_TOP_K", 5)
DEFAULT_SEARCH_METHOD = os.getenv("DEFAULT_SEARCH_METHOD", "auto")  # auto|vector|keyword|hybrid

# Seuils de score en dessous desquels on considère qu'aucune réponse fiable
# n'a été trouvée dans les documents (voir agent/context_builder.py).
# Chaque méthode a sa propre échelle de score (voir search/search_utils.py de
# Membre 2) : ces seuils ne sont donc PAS comparables entre eux.
SEUIL_VECTOR = float(os.getenv("SEUIL_NO_ANSWER_VECTOR", "0.12"))
SEUIL_KEYWORD = float(os.getenv("SEUIL_NO_ANSWER_KEYWORD", "1"))
SEUIL_HYBRID = float(os.getenv("SEUIL_NO_ANSWER_HYBRID", "0.0005"))

MAX_TOOL_CALLS_AGENT = _get_int("MAX_TOOL_CALLS_AGENT", 4)

