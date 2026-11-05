"""Recherche hybride : combine vector_search et keyword_search par fusion de rang
(Reciprocal Rank Fusion, RRF).

Pourquoi RRF et pas une simple moyenne pondérée des scores ?
--------------------------------------------------------------
Le score de vector_search (similarité cosinus, ~[-1, 1]) et le score de
keyword_search (SCORE Oracle Text, un entier ~[0, 100] sans borne garantie) ne
sont PAS sur la même échelle et ne sont pas directement comparables ni
additionnables sans normalisation arbitraire. RRF évite ce problème : il
n'utilise que le RANG (1er, 2e, 3e...) de chaque document dans chaque liste,
pas sa valeur de score brute. C'est la méthode utilisée en pratique par la
plupart des moteurs de recherche hybrides (ex. Elasticsearch, Azure AI Search).

Formule, pour un document d présent au rang r_v dans la liste vectorielle
et/ou au rang r_k dans la liste mots-clés :

    rrf_score(d) = w_vector * 1 / (k + r_v) + w_keyword * 1 / (k + r_k)

(le terme correspondant est omis si le document n'apparaît pas dans cette liste)

- k (rrf_k, défaut 60) est la constante usuelle de la littérature RRF : elle
  amortit l'écart entre les tout premiers rangs, pour qu'un document 1er dans
  une seule liste n'écrase pas un document 2e dans les deux listes.
- w_vector / w_keyword permettent de favoriser un mode par rapport à l'autre
  (défaut 1.0 / 1.0, c'est-à-dire aucune préférence).

Limites assumées (à ne pas sur-vendre en soutenance) :
- RRF ignore l'intensité du score, seulement le rang : un document 1er avec une
  similarité de 0.95 compte comme un document 1er avec une similarité de 0.51.
- La qualité dépend entièrement du nombre de candidats récupérés par méthode
  (candidate_k) : un document pertinent absent des deux listes de candidats ne
  peut pas remonter par fusion.
- Aucun reranking par cross-encoder n'est appliqué : seule la fusion de rang
  est implémentée ici.
"""

from __future__ import annotations

import logging
from typing import Optional

import oracledb

from .keyword_search import keyword_search
from .search_utils import SearchResult, get_connection, valider_requete, valider_top_k
from .vector_search import vector_search

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def hybrid_search(
    query: str,
    top_k: int = 5,
    *,
    candidate_k: Optional[int] = None,
    rrf_k: int = DEFAULT_RRF_K,
    w_vector: float = 1.0,
    w_keyword: float = 1.0,
    connection: Optional[oracledb.Connection] = None,
) -> list[SearchResult]:
    """Recherche hybride (vectorielle + mots-clés) fusionnée par RRF.

    Args:
        query: question ou phrase en langage naturel.
        top_k: nombre de résultats finaux à retourner (défaut 5).
        candidate_k: nombre de candidats récupérés PAR méthode avant fusion
            (défaut : max(20, top_k * 4), pour donner à la fusion suffisamment
            de matière ; augmenter si le rappel semble insuffisant).
        rrf_k: constante RRF (défaut 60, valeur usuelle de la littérature).
        w_vector / w_keyword: poids relatifs des deux méthodes dans le score
            final (défaut 1.0 chacun = aucune préférence).
        connection: connexion Oracle déjà ouverte, optionnelle. Si fournie, elle
            est réutilisée pour les deux sous-recherches (une seule connexion
            au lieu de deux).

    Returns:
        Liste de SearchResult (search_method="hybrid") triée par score RRF
        décroissant. Chaque résultat garde, dans son champ metadata, les scores
        et rangs d'origine (vector_score/vector_rank, keyword_score/keyword_rank)
        quand ils existent, pour audit/debug.

    Raises:
        SearchError: propagée telle quelle depuis vector_search / keyword_search
            (requête invalide, erreur Oracle, index manquant...).
    """
    query = valider_requete(query)
    top_k = valider_top_k(top_k)
    if candidate_k is None:
        candidate_k = max(20, top_k * 4)
    candidate_k = valider_top_k(candidate_k)

    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        resultats_vecteur = vector_search(query, top_k=candidate_k, connection=conn)
        resultats_motscles = keyword_search(query, top_k=candidate_k, connection=conn)
    finally:
        if owns_connection:
            conn.close()

    # rang 1-indexé dans chaque liste
    rang_vecteur = {r.chunk_id: (i + 1, r) for i, r in enumerate(resultats_vecteur)}
    rang_motscles = {r.chunk_id: (i + 1, r) for i, r in enumerate(resultats_motscles)}

    tous_les_ids = set(rang_vecteur) | set(rang_motscles)

    fusionnes: list[SearchResult] = []
    for chunk_id in tous_les_ids:
        rrf_score = 0.0
        extra_meta: dict = {}
        base_result: Optional[SearchResult] = None

        if chunk_id in rang_vecteur:
            rang, r = rang_vecteur[chunk_id]
            rrf_score += w_vector * (1.0 / (rrf_k + rang))
            extra_meta["vector_score"] = r.score
            extra_meta["vector_rank"] = rang
            base_result = r

        if chunk_id in rang_motscles:
            rang, r = rang_motscles[chunk_id]
            rrf_score += w_keyword * (1.0 / (rrf_k + rang))
            extra_meta["keyword_score"] = r.score
            extra_meta["keyword_rank"] = rang
            if base_result is None:
                base_result = r

        assert base_result is not None  # toujours vrai : chunk_id vient d'une des 2 listes

        fusionnes.append(
            SearchResult(
                chunk_id=base_result.chunk_id,
                document_id=base_result.document_id,
                content=base_result.content,
                source=base_result.source,
                score=rrf_score,
                search_method="hybrid",
                metadata={**base_result.metadata, **extra_meta, "rrf_k": rrf_k},
            )
        )

    fusionnes.sort(key=lambda r: r.score, reverse=True)
    return fusionnes[:top_k]


if __name__ == "__main__":
    # Exemple d'utilisation — adapté au corpus réel du projet (documents réseaux).
    for r in hybrid_search("différence entre TCP et UDP", top_k=3):
        meta = r.metadata
        print(
            f"[rrf={r.score:.4f} | v_rank={meta.get('vector_rank')} "
            f"k_rank={meta.get('keyword_rank')}] {r.source} -> {r.content[:100]}..."
        )
