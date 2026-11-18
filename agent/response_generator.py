"""Génération de la réponse finale et mise au format JSON standardisé
communiqué à Streamlit (Membre 4).

Séparé de agent.py pour pouvoir être testé seul, avec un LLM factice, sans
dépendre de LangGraph ni d'une clé API (voir tests/test_responses.py).
"""

from __future__ import annotations

from typing import Any

from .context_builder import build_context, collect_sources, detect_ambiguous, detect_no_answer
from .prompts import AMBIGUOUS_PREFIX, NO_ANSWER_MESSAGE, RESPONSE_PROMPT_TEMPLATE


def generate_answer(
    llm: Any,
    question: str,
    results: list[dict[str, Any]],
    *,
    no_answer: bool | None = None,
    ambiguous: bool | None = None,
) -> str:
    """Produit le texte de réponse final.

    Si `no_answer` (ou sa détection automatique) est vrai, aucune question
    n'est posée au LLM : on renvoie directement le message standard — plus
    rapide, moins coûteux, et garanti sans hallucination.
    """
    if no_answer is None:
        no_answer = detect_no_answer(results)
    if no_answer:
        return NO_ANSWER_MESSAGE

    if ambiguous is None:
        ambiguous = detect_ambiguous(results)

    context = build_context(results)
    prompt = RESPONSE_PROMPT_TEMPLATE.format(context=context, question=question)
    reponse = llm.invoke(prompt)
    texte = getattr(reponse, "content", None) or str(reponse)

    if ambiguous:
        texte = AMBIGUOUS_PREFIX + texte
    return texte


def _garder_sources_citees(sources: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
    """Si la réponse cite explicitement des documents (format [source : fichier]),
    on n'affiche que ceux-là : les autres passages récupérés par la recherche
    mais non utilisés par la réponse ne sont pas de vraies sources. Si aucun
    document n'est cité dans le texte, on garde toute la liste."""
    if not sources or not answer:
        return sources
    texte = answer.lower()
    citees = [
        s for s in sources
        if s.get("document_id") and str(s["document_id"]).lower() in texte
    ]
    return citees or sources


def build_response_payload(
    *,
    question: str,
    answer: str,
    results: list[dict[str, Any]],
    search_method: str,
    error: str | None = None,
    no_answer: bool | None = None,
    ambiguous: bool | None = None,
) -> dict[str, Any]:
    """Construit le format JSON standardisé attendu par Streamlit (Membre 4).

    Schéma garanti stable (toujours les mêmes clés, même en cas d'erreur) :
        {
          "question": str,
          "answer": str,
          "sources": [{"source", "document_id", "score", "search_method"}],
          "search_method": str,           # méthode(s) effectivement utilisée(s)
          "no_answer_found": bool,
          "ambiguous": bool,
          "error": str | null,
          "raw_results": [...]            # résultats bruts, pour debug/affichage avancé
        }
    """
    if no_answer is None:
        no_answer = detect_no_answer(results) if results is not None else True
    if ambiguous is None:
        ambiguous = detect_ambiguous(results) if results else False
    if no_answer:
        ambiguous = False  # une absence de réponse n'est pas une ambiguïté

    # Pas de sources affichées quand aucune réponse n'a été trouvée.
    sources = [] if no_answer or not results else collect_sources(results)
    sources = _garder_sources_citees(sources, answer)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "search_method": search_method,
        "no_answer_found": bool(no_answer),
        "ambiguous": bool(ambiguous),
        "error": error,
        "raw_results": results or [],
    }