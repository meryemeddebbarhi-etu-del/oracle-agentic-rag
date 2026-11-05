"""Recherche vectorielle (sémantique) sur la table créée par Membre 1.

Réutilise :
- la même table (DOCUMENTS_RAG, via search_utils.TABLE_NAME)
- le même modèle d'embeddings (search_utils.get_embeddings(), = Membre 1)
- le même index vectoriel que Membre 1 a créé (HNSW, ou IVF en repli) :
  on ne recrée rien ici, on se contente d'interroger la colonne "embedding".

La requête SQL suit le même principe que le notebook du professeur (section 7) :
VECTOR_DISTANCE(embedding, :q, COSINE) avec FETCH APPROX FIRST n ROWS ONLY,
qui laisse l'optimiseur Oracle utiliser l'index HNSW/IVF s'il existe.
"""

from __future__ import annotations

import array
import logging
from typing import Optional

import oracledb

from .search_utils import (
    SearchError,
    SearchResult,
    TABLE_NAME,
    formater_source,
    get_connection,
    get_embeddings,
    lire_clob,
    lire_metadata_json,
    valider_requete,
    valider_top_k,
)

logger = logging.getLogger(__name__)


def vector_search(
    query: str,
    top_k: int = 5,
    *,
    connection: Optional[oracledb.Connection] = None,
) -> list[SearchResult]:
    """Recherche sémantique (par similarité vectorielle).

    Args:
        query: question ou phrase en langage naturel.
        top_k: nombre de résultats à retourner (défaut 5).
        connection: connexion Oracle déjà ouverte, optionnelle. Si absente,
            une connexion est ouverte puis fermée pour cet appel (pratique
            pour Membre 3, qui n'a pas à gérer les connexions lui-même).

    Returns:
        Liste de SearchResult triée par pertinence décroissante
        (score = 1 - distance cosinus, donc 1.0 = identique, proche de 0 = sans
        rapport ; peut être légèrement négatif pour des vecteurs opposés, ce qui
        est normal mathématiquement et ne doit pas être traité comme une erreur).

    Raises:
        SearchError: requête vide/invalide, top_k invalide, erreur Oracle,
            ou colonne/index vectoriel absent de la table.
    """
    query = valider_requete(query)
    top_k = valider_top_k(top_k)

    embeddings = get_embeddings()
    query_vector = array.array("f", embeddings.embed_query(query))

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
                       VECTOR_DISTANCE(embedding, :q, COSINE) AS distance
                FROM {TABLE_NAME}
                ORDER BY distance
                FETCH APPROX FIRST {top_k} ROWS ONLY
                """,
                q=query_vector,
            )
            rows = cursor.fetchall()
        except oracledb.DatabaseError as erreur:
            raise SearchError(
                f"Échec de la recherche vectorielle sur {TABLE_NAME} "
                f"(colonne 'embedding' ou index vectoriel manquant ?) : {erreur}"
            ) from erreur
        finally:
            cursor.close()
    finally:
        if owns_connection:
            conn.close()

    resultats: list[SearchResult] = []
    for row_id, texte, metadata_brute, distance in rows:
        metadata = lire_metadata_json(metadata_brute)
        contenu = lire_clob(texte)
        score = 1.0 - float(distance)  # inverse exact de la distance cosinus
        resultats.append(
            SearchResult(
                chunk_id=row_id,
                document_id=str(metadata.get("source", "inconnu")),
                content=contenu,
                source=formater_source(metadata),
                score=score,
                search_method="vector",
                metadata={**metadata, "distance": float(distance)},
            )
        )
    return resultats


if __name__ == "__main__":
    # Exemple d'utilisation — adapté au corpus réel du projet (documents réseaux).
    for r in vector_search("comment fonctionne la résolution de noms DNS", top_k=3):
        print(f"[{r.score:.4f}] {r.source} -> {r.content[:100]}...")
