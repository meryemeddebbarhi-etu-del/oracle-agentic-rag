"""Construction du contexte documentaire à partir des résultats de recherche,
et détection des cas particuliers demandés par le cahier des charges :
absence de réponse dans les documents, question ambiguë (plusieurs passages
pertinents de pertinence proche).

Ce module ne connaît ni le LLM ni LangGraph : il ne travaille que sur les
listes de dict renvoyées par SearchAdapter (ou par ToolCallRecorder), ce qui
le rend testable indépendamment du reste (voir tests/test_responses.py).
"""

from __future__ import annotations

from typing import Any

from config import settings

# Seuils par méthode : les échelles de score ne sont PAS comparables entre
# vector (cosinus, environ [-1, 1]), keyword (score Oracle Text, entier non
# borné) et hybrid (score RRF, très petit par construction).
_SEUILS = {
    "vector": settings.SEUIL_VECTOR,
    "keyword": settings.SEUIL_KEYWORD,
    "hybrid": settings.SEUIL_HYBRID,
}

# Écart relatif en dessous duquel on considère les deux meilleurs résultats
# comme "de pertinence proche" (question potentiellement ambiguë).
_ECART_AMBIGU = 0.15


def build_context(results: list[dict[str, Any]], max_chars: int = 3000) -> str:
    """Concatène les passages en un contexte lisible par le LLM, borné en
    taille pour ne pas dépasser la fenêtre de contexte du modèle."""
    if not results:
        return ""
    blocs = []
    taille = 0
    vus: set[str] = set()
    for r in results:
        chunk_id = r.get("chunk_id")
        if chunk_id is not None and chunk_id in vus:
            continue
        if chunk_id is not None:
            vus.add(chunk_id)
        bloc = f"[Source : {r.get('source', 'inconnue')}]\n{r.get('content', '').strip()}"
        if taille + len(bloc) > max_chars and blocs:
            break
        blocs.append(bloc)
        taille += len(bloc)
    return "\n\n---\n\n".join(blocs)


def collect_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retourne les sources uniques dans l'ordre de récupération.

    Les scores de méthodes différentes ne sont pas comparables; la première
    occurrence de chaque source conserve sa méthode et son score d'origine.
    """
    vues: dict[str, dict[str, Any]] = {}
    for r in results:
        source = r.get("source", "inconnue")
        if source not in vues:
            vues[source] = {
                "source": source,
                "document_id": r.get("document_id"),
                "score": r.get("score", 0),
                "search_method": r.get("search_method"),
            }
    return list(vues.values())


def detect_no_answer(results: list[dict[str, Any]], method: str | None = None) -> bool:
    """Détecte les questions sans réponse disponible dans les documents
    (point 9 du cahier des charges) : aucun résultat, ou tous les résultats
    sont sous le seuil de pertinence de leur méthode.

    Pour l'hybride réel, le score RRF ne dépend que des rangs : on juge donc
    sur les scores d'origine (vector_score / keyword_score) gardés en metadata.
    """
    if not results:
        return True
    for r in results:
        m = r.get("search_method", method or "hybrid")
        meta = r.get("metadata") or {}
        if m == "hybrid" and ("vector_score" in meta or "keyword_score" in meta):
            if (
                meta.get("vector_score", float("-inf")) >= _SEUILS["vector"]
                or meta.get("keyword_score", float("-inf")) >= _SEUILS["keyword"]
            ):
                return False
        elif r.get("score", 0) >= _SEUILS.get(m, 0):
            return False
    return True


def detect_ambiguous(results: list[dict[str, Any]]) -> bool:
    """Détecte une question potentiellement ambiguë : les DEUX meilleurs
    résultats proviennent de documents DIFFÉRENTS et ont un score très
    proche l'un de l'autre (on ne compare que le top-2, pas toute la liste,
    pour ne pas déclencher le signal dès qu'un corpus a naturellement
    plusieurs scores voisins).

    La méthode "hybrid" est ignorée : son score RRF ne mesure que des rangs,
    donc deux documents différents ont toujours des scores voisins, ce qui
    ne prouve aucune ambiguïté. Seuls vector et keyword sont jugés.
    """
    par_methode: dict[str, list[dict[str, Any]]] = {}
    for resultat in results:
        method = resultat.get("search_method", "hybrid")
        par_methode.setdefault(method, []).append(resultat)

    for methode, resultats in par_methode.items():
        if methode == "hybrid":
            continue
        if len(resultats) < 2:
            continue
        premier, second = sorted(
            resultats, key=lambda r: r.get("score", 0), reverse=True
        )[:2]
        if premier.get("document_id") == second.get("document_id"):
            continue
        ecart = abs(premier.get("score", 0) - second.get("score", 0))
        reference = max(abs(premier.get("score", 0)), 1e-9)
        if ecart <= _ECART_AMBIGU * reference:
            return True
    return False