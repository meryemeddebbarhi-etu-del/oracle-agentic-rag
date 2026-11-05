"""Recherche par mots-clés avec Oracle Text, sur l'index déjà créé par Membre 1.

IMPORTANT : ce module NE CRÉE PAS d'index. L'index Oracle Text
(DOCUMENTS_RAG_TEXT_IDX, type CTXSYS.CONTEXT) est déjà créé par
database/ingestion.py (fonction creer_index). Le recréer ici dupliquerait le
travail de Membre 1 et provoquerait une erreur Oracle (index déjà existant sur
cette colonne). On se contente d'utiliser CONTAINS(text, ...), qui s'appuie
automatiquement sur cet index.

Si l'index n'existe pas encore (ingestion.py pas encore exécuté, ou création de
l'index échouée), la requête Oracle échoue avec l'erreur DRG-10599 ("index Oracle
Text manquant sur cette colonne") — ce module la détecte et renvoie un message
explicite plutôt qu'une trace Oracle brute.
"""

from __future__ import annotations

import logging
from typing import Optional

import oracledb

from .search_utils import (
    SearchError,
    SearchResult,
    TABLE_NAME,
    construire_requete_contains,
    formater_source,
    get_connection,
    lire_clob,
    lire_metadata_json,
    valider_requete,
    valider_top_k,
)

logger = logging.getLogger(__name__)

# Index créé par database/ingestion.py::creer_index() — utilisé seulement pour
# produire un message d'erreur clair ; CONTAINS n'a pas besoin de son nom.
NOM_INDEX_TEXTE = "DOCUMENTS_RAG_TEXT_IDX"


def keyword_search(
    query: str,
    top_k: int = 5,
    *,
    fuzzy: bool = False,
    connection: Optional[oracledb.Connection] = None,
) -> list[SearchResult]:
    """Recherche par mots-clés exacts (Oracle Text / CONTAINS).

    Args:
        query: mots-clés ou phrase. Chaque mot est recherché indépendamment et
            combiné avec l'opérateur ACCUM (voir search_utils.construire_requete_contains) :
            un document contenant plusieurs des mots recherchés obtient un score
            plus élevé, mais un document n'a pas besoin de contenir TOUS les mots
            pour apparaître dans les résultats.
        top_k: nombre de résultats à retourner (défaut 5).
        fuzzy: si True, tolère les fautes de frappe (FUZZY(terme) au lieu de "terme").
        connection: connexion Oracle déjà ouverte, optionnelle.

    Returns:
        Liste de SearchResult triée par score Oracle Text (SCORE(1)) décroissant.
        Le score Oracle Text est un entier entre 0 et 100 (pas une probabilité),
        propre à Oracle Text : il n'est PAS comparable directement au score de
        vector_search (voir hybrid_search pour la fusion correcte des deux).

    Raises:
        SearchError: requête vide, top_k invalide, erreur Oracle Text (y compris
            index Oracle Text manquant), ou tout autre échec de la requête.
    """
    query = valider_requete(query)
    top_k = valider_top_k(top_k)

    expression = construire_requete_contains(query, fuzzy=fuzzy)
    if not expression:
        # La requête ne contenait aucun mot exploitable (ex. uniquement de la
        # ponctuation) : pas d'erreur, juste aucun résultat.
        return []

    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"""
                SELECT RAWTOHEX(id) AS id,
                       text,
                       metadata,
                       SCORE(1) AS score
                FROM {TABLE_NAME}
                WHERE CONTAINS(text, :expression, 1) > 0
                ORDER BY score DESC
                FETCH FIRST {top_k} ROWS ONLY
                """,
                expression=expression,
            )
            rows = cursor.fetchall()
        except oracledb.DatabaseError as erreur:
            message = str(erreur)
            if "DRG-10599" in message or "DRG-10700" in message:
                raise SearchError(
                    f"L'index Oracle Text '{NOM_INDEX_TEXTE}' est absent de la "
                    f"table {TABLE_NAME}. Exécutez d'abord database/ingestion.py "
                    "(il crée cet index automatiquement), ou créez-le manuellement :\n"
                    f"  CREATE INDEX {NOM_INDEX_TEXTE} ON {TABLE_NAME}(text) "
                    "INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('SYNC (ON COMMIT)');"
                ) from erreur
            raise SearchError(f"Échec de la recherche par mots-clés : {erreur}") from erreur
        finally:
            cursor.close()
    finally:
        if owns_connection:
            conn.close()

    resultats: list[SearchResult] = []
    for row_id, texte, metadata_brute, score in rows:
        metadata = lire_metadata_json(metadata_brute)
        contenu = lire_clob(texte)
        resultats.append(
            SearchResult(
                chunk_id=row_id,
                document_id=str(metadata.get("source", "inconnu")),
                content=contenu,
                source=formater_source(metadata),
                score=float(score),
                search_method="keyword",
                metadata={**metadata, "oracle_text_score": float(score)},
            )
        )
    return resultats


if __name__ == "__main__":
    # Exemple d'utilisation — adapté au corpus réel du projet (documents réseaux).
    for r in keyword_search("DHCP", top_k=3):
        print(f"[{r.score:.1f}] {r.source} -> {r.content[:100]}...")
