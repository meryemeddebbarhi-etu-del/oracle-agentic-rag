"""Outils partagés par les trois modes de recherche (vectorielle, mots-clés, hybride).

Ce module ne fait QUE réutiliser ce que le Membre 1 a déjà mis en place :
- la même connexion Oracle (mêmes variables d'environnement, database/.env)
- le même modèle d'embeddings (database.ingestion.MODELE_EMBEDDING)
- la même table (database.ingestion.NOM_TABLE)
- le même schéma de colonnes créé par OracleVS : id (RAW16), text (CLOB),
  metadata (JSON), embedding (VECTOR)

Rien n'est codé en dur ici : si Membre 1 change le nom de la table ou le modèle
d'embeddings dans database/ingestion.py, ce module suit automatiquement.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import oracledb
from dotenv import load_dotenv

# Ces deux constantes DOIVENT rester identiques à database/ingestion.py
# (NOM_TABLE et MODELE_EMBEDDING). Elles ne sont volontairement PAS importées
# depuis ce module : database/ingestion.py importe aussi pypdf et
# langchain_text_splitters (lecture/découpage de documents), qui n'ont rien à
# voir avec la recherche. Les importer ici forcerait tout le module "search"
# (donc l'agent de Membre 3) à dépendre de ces bibliothèques de lecture de
# documents, uniquement pour lire deux chaînes de caractères. Si Membre 1
# change le nom de la table ou le modèle d'embeddings, ces deux lignes doivent
# être mises à jour en même temps (un test de cohérence est fourni dans
# tests/test_search_utils.py::test_table_name_coherente_avec_ingestion).
TABLE_NAME = "DOCUMENTS_RAG"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

# Cache module-level pour ne charger le modèle d'embeddings qu'une seule fois
# par processus (il est coûteux à charger : ~130 Mo, 1-3 s).
_embeddings_singleton = None


class SearchError(Exception):
    """Erreur de recherche : connexion Oracle, requête invalide, index manquant..."""


def get_connection() -> oracledb.Connection:
    """Ouvre une connexion Oracle avec les mêmes variables d'environnement que Membre 1.

    Lève SearchError (et pas une exception oracledb brute) pour que Membre 3
    puisse l'attraper facilement sans connaître oracledb.
    """
    load_dotenv()
    user = os.getenv("ORACLE_USER")
    pwd = os.getenv("ORACLE_PWD")
    dsn = os.getenv("ORACLE_DSN")
    if not user or not pwd or not dsn:
        raise SearchError(
            "Variables d'environnement Oracle manquantes "
            "(ORACLE_USER / ORACLE_PWD / ORACLE_DSN). Vérifiez le fichier .env."
        )
    timeout = int(os.getenv("ORACLE_CONNECT_TIMEOUT", "5"))
    try:
        return oracledb.connect(
            user=user, password=pwd, dsn=dsn, tcp_connect_timeout=timeout
        )
    except oracledb.DatabaseError as erreur:
        raise SearchError(f"Connexion à Oracle impossible : {erreur}") from erreur


def get_embeddings():
    """Retourne l'objet d'embeddings HuggingFace, en le créant une seule fois.

    Utilise exactement le modèle de Membre 1 (database.ingestion.MODELE_EMBEDDING),
    donc les vecteurs de requête sont dans le même espace que les vecteurs stockés.
    """
    global _embeddings_singleton
    if _embeddings_singleton is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings_singleton = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_singleton


def valider_requete(query: str) -> str:
    """Valide et nettoie une requête texte. Lève SearchError si elle est vide."""
    if query is None:
        raise SearchError("La requête ne peut pas être vide (None).")
    query = query.strip()
    if not query:
        raise SearchError("La requête ne peut pas être vide.")
    return query


def valider_top_k(top_k: int) -> int:
    """Valide top_k : doit être un entier strictement positif, borné pour éviter
    un FETCH FIRST déraisonnable (top_k est injecté dans le SQL en littéral, donc
    on le valide strictement ici — jamais de chaîne venant de l'utilisateur)."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise SearchError("top_k doit être un entier.")
    if top_k < 1:
        raise SearchError("top_k doit être supérieur ou égal à 1.")
    if top_k > 200:
        raise SearchError("top_k est trop grand (maximum 200).")
    return top_k


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def construire_requete_contains(query: str, fuzzy: bool = False) -> str:
    """Transforme une requête en langage naturel en expression Oracle Text.

    Oracle Text (CONTAINS) n'accepte pas n'importe quelle chaîne : la ponctuation,
    les accents collés à des opérateurs, etc. peuvent provoquer une erreur DRG-xxxxx.
    On découpe donc la requête en mots, on met chaque mot entre guillemets (ou dans
    FUZZY(...) si fuzzy=True pour tolérer les fautes de frappe), et on les combine
    avec ACCUM : chaque terme présent augmente le score du document, sans exiger
    que tous les termes soient présents (contrairement à AND).

    Exemple : "protocole DNS"  ->  '"protocole" ACCUM "DNS"'
    """
    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return ""
    if fuzzy:
        parts = [f'FUZZY("{tok}")' for tok in tokens]
    else:
        parts = [f'"{tok}"' for tok in tokens]
    return " ACCUM ".join(parts)


@dataclass
class SearchResult:
    """Format de résultat unique et standardisé, le même pour les 3 méthodes de
    recherche. C'est l'interface que Membre 3 doit utiliser : il n'a jamais besoin
    de connaître le SQL sous-jacent.

    Champs :
      chunk_id      identifiant unique du morceau (= colonne "id" de la table,
                    en hexadécimal — garanti unique par morceau)
      document_id   identifiant du document source. Le schéma de Membre 1 ne
                    contient pas d'identifiant de document séparé : c'est donc
                    le nom de fichier (metadata["source"]).
      content       le texte du morceau
      source        chaîne lisible pour la citation, ex. "fichier.pdf (p. 3)"
      score         score de pertinence, DANS L'ÉCHELLE DE LA MÉTHODE QUI L'A
                    PRODUIT (voir search_method) — ne jamais comparer un score
                    "vector" brut à un score "keyword" brut, ce ne sont pas la
                    même échelle (voir hybrid_search pour la fusion correcte)
      search_method "vector" | "keyword" | "hybrid"
      metadata      toutes les métadonnées brutes de Membre 1 (page, categorie,
                    chunk_id interne) + les détails spécifiques à la méthode
                    (ex. vector_score / keyword_score / rank pour le mode hybride)
    """

    chunk_id: str
    document_id: str
    content: str
    source: str
    score: float
    search_method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "search_method": self.search_method,
            "metadata": self.metadata,
        }


def formater_source(metadata: dict[str, Any]) -> str:
    """Construit une chaîne de citation lisible à partir des métadonnées de Membre 1."""
    nom = metadata.get("source", "document inconnu")
    page = metadata.get("page")
    if page and str(nom).lower().endswith(".pdf"):
        return f"{nom} (p. {page})"
    return str(nom)


def lire_metadata_json(valeur: Any) -> dict[str, Any]:
    """La colonne JSON peut revenir comme dict (déjà décodé par oracledb) ou comme
    chaîne selon la configuration du driver : on gère les deux cas sans supposer."""
    if valeur is None:
        return {}
    if isinstance(valeur, dict):
        return valeur
    if isinstance(valeur, (bytes, bytearray)):
        valeur = valeur.decode("utf-8", errors="ignore")
    if isinstance(valeur, str):
        import json

        try:
            return json.loads(valeur)
        except json.JSONDecodeError:
            return {}
    return {}


def lire_clob(valeur: Any) -> str:
    """Lit un CLOB oracledb (ou une chaîne déjà matérialisée) en texte Python."""
    if valeur is None:
        return ""
    if hasattr(valeur, "read"):
        return valeur.read()
    return str(valeur)
