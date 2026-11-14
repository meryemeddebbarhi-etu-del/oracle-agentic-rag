"""Outils LangChain/LangGraph exposés au LLM pour qu'il choisisse lui-même la
méthode de recherche (vector / keyword / hybrid), conformément au point 2 du
cahier des charges.

Chaque outil :
  - appelle le SearchAdapter (jamais directement le moteur de Membre 2) ;
  - attrape SearchError et renvoie un message texte explicite au LLM au lieu
    de faire planter l'agent (point 10 : gérer les erreurs du moteur) ;
  - enregistre les résultats bruts (liste de dict) dans un ToolCallRecorder,
    pour que agent.py puisse reconstruire le contexte, les sources et
    détecter l'absence de réponse SANS avoir à re-parser le texte que le LLM
    a vu (plus robuste qu'une extraction depuis la sortie du modèle).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool

from agent_search import SearchAdapter, SearchError


@dataclass
class ToolCallRecorder:
    """Historique des appels aux outils de recherche effectués pendant UNE
    invocation de l'agent. Remis à zéro avant chaque question (voir agent.py).
    """

    history: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def enregistrer(self, method: str, query: str, results: list[dict[str, Any]]) -> None:
        self.history.append({"method": method, "query": query, "results": results})

    def tous_les_resultats(self) -> list[dict[str, Any]]:
        """Fusionne les appels en comparant les scores seulement dans une
        même méthode : vector, Oracle Text et RRF ont des échelles distinctes."""
        par_methode: dict[str, dict[str, dict[str, Any]]] = {}
        for appel in self.history:
            method = appel["method"]
            meilleurs = par_methode.setdefault(method, {})
            for resultat in appel["results"]:
                cid = resultat.get("chunk_id")
                if cid is None:
                    cid = f"sans_id_{len(meilleurs)}"
                if cid not in meilleurs or resultat.get("score", 0) > meilleurs[cid].get("score", 0):
                    meilleurs[cid] = resultat

        fusionnes: list[dict[str, Any]] = []
        for resultats in par_methode.values():
            fusionnes.extend(sorted(resultats.values(), key=lambda r: r.get("score", 0), reverse=True))
        return fusionnes

    def methodes_utilisees(self) -> list[str]:
        vues: list[str] = []
        for appel in self.history:
            if appel["method"] not in vues:
                vues.append(appel["method"])
        return vues


def _formater_pour_llm(method: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return f"Aucun résultat trouvé par la recherche {method} pour cette requête."
    blocs = []
    for r in results:
        blocs.append(
            json.dumps(
                {
                    "source": r.get("source"),
                    "score": r.get("score"),
                    "extrait": (r.get("content") or "")[:600],
                },
                ensure_ascii=False,
            )
        )
    return f"{len(results)} résultat(s) ({method}) :\n" + "\n".join(blocs)


def create_search_tools(adapter: SearchAdapter, recorder: ToolCallRecorder, top_k: int = 5) -> list:
    """Construit les 3 outils de recherche liés à un adapter et un recorder
    donnés (nécessaire car @tool ne peut pas prendre d'argument supplémentaire
    au moment où le LLM l'appelle : on referme sur `adapter`/`recorder` via
    des closures)."""

    @tool
    def search_vector(query: str) -> str:
        """Recherche sémantique (par similarité de sens) dans les documents.
        Utile pour les questions formulées librement ou conceptuelles."""
        try:
            resultats = adapter.vector_search(query, top_k=top_k)
        except SearchError as erreur:
            return f"Erreur du moteur de recherche (vector) : {erreur}"
        recorder.enregistrer("vector", query, resultats)
        return _formater_pour_llm("vector", resultats)

    @tool
    def search_keyword(query: str) -> str:
        """Recherche par mots-clés exacts (noms de protocoles, acronymes,
        termes techniques précis) dans les documents."""
        try:
            resultats = adapter.keyword_search(query, top_k=top_k)
        except SearchError as erreur:
            return f"Erreur du moteur de recherche (keyword) : {erreur}"
        recorder.enregistrer("keyword", query, resultats)
        return _formater_pour_llm("keyword", resultats)

    @tool
    def search_hybrid(query: str) -> str:
        """Recherche hybride combinant sens et mots-clés. À utiliser par
        défaut en cas de doute, ou si vector/keyword n'ont rien donné."""
        try:
            resultats = adapter.hybrid_search(query, top_k=top_k)
        except SearchError as erreur:
            return f"Erreur du moteur de recherche (hybrid) : {erreur}"
        recorder.enregistrer("hybrid", query, resultats)
        return _formater_pour_llm("hybrid", resultats)

    return [search_vector, search_keyword, search_hybrid]

