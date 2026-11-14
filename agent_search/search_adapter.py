"""SearchAdapter — point d'intégration unique entre l'agent (Membre 3) et le
moteur de recherche (Membre 2).

L'agent ne doit JAMAIS appeler directement `search.vector_search(...)` ou
`mock_search.vector_search(...)` : il passe toujours par un objet
`SearchAdapter`, qui expose trois méthodes identiques quel que soit le
backend réel derrière :

    adapter.vector_search(query, top_k)
    adapter.keyword_search(query, top_k)
    adapter.hybrid_search(query, top_k)

Chacune renvoie une liste de dict avec exactement les clés produites par
Membre 2 (SearchResult.to_dict()) : chunk_id, document_id, content, source,
score, search_method, metadata.

Pour brancher le vrai moteur de Membre 2 quand il est prêt :
    1. Copier le dossier `search/` de Membre 2 (celui contenant
       search_utils.py, vector_search.py, keyword_search.py,
       hybrid_search.py) à la racine du projet, à côté de ce fichier — ou
       l'installer comme dépendance du projet.
    2. Mettre SEARCH_BACKEND=real dans le fichier .env.
    3. Renseigner ORACLE_USER / ORACLE_PWD / ORACLE_DSN dans le même .env
       (ce sont les variables que lit le module de Membre 2).
Rien d'autre ne change : agent/, tests/ et main.py utilisent l'adapter, pas
le backend directement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from . import mock_search
from .errors import SearchError

logger = logging.getLogger(__name__)

__all__ = ["SearchAdapter", "MockSearchAdapter", "RealSearchAdapter", "SearchError", "get_search_adapter"]


class SearchAdapter(ABC):
    """Interface commune : toute implémentation doit fournir ces trois méthodes."""

    @abstractmethod
    def vector_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def keyword_search(self, query: str, top_k: int = 5, *, fuzzy: bool = False) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...

    def search(self, query: str, top_k: int = 5, method: str = "hybrid") -> list[dict[str, Any]]:
        """Point d'entrée générique utilisé par l'agent quand la méthode est
        choisie dynamiquement (par l'utilisateur ou par le LLM)."""
        method = (method or "hybrid").lower()
        if method == "vector":
            return self.vector_search(query, top_k)
        if method == "keyword":
            return self.keyword_search(query, top_k)
        if method == "hybrid":
            return self.hybrid_search(query, top_k)
        raise SearchError(
            f"Méthode de recherche inconnue : '{method}'. "
            "Valeurs acceptées : vector, keyword, hybrid."
        )


class MockSearchAdapter(SearchAdapter):
    """Backend simulé (aucune dépendance Oracle) — utilisé par défaut, et dans
    tous les tests automatisés, pour que l'agent soit développable et
    démontrable indépendamment de l'avancement du Membre 2."""

    def vector_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return mock_search.vector_search(query, top_k=top_k)

    def keyword_search(self, query: str, top_k: int = 5, *, fuzzy: bool = False) -> list[dict[str, Any]]:
        return mock_search.keyword_search(query, top_k=top_k, fuzzy=fuzzy)

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return mock_search.hybrid_search(query, top_k=top_k)


class RealSearchAdapter(SearchAdapter):
    """Backend réel — délègue au module `search` du Membre 2.

    L'import est fait ici (pas en haut du fichier) pour que le reste du
    projet (agent, tests, MockSearchAdapter) continue de fonctionner même si
    le module de Membre 2, Oracle, ou les dépendances lourdes (oracledb,
    sentence-transformers) ne sont pas installés.
    """

    def __init__(self) -> None:
        try:
            import search as moteur_membre2  # type: ignore
        except ImportError as erreur:
            raise SearchError(
                "Le vrai moteur de recherche (module 'search' du Membre 2) est "
                "introuvable. Placez son dossier search/ à la racine du projet "
                "et installez ses dépendances (oracledb, langchain-huggingface), "
                "ou repassez SEARCH_BACKEND=mock dans .env. "
                f"Erreur d'import d'origine : {erreur}"
            ) from erreur
        self._moteur = moteur_membre2
        self._SearchErrorMembre2 = moteur_membre2.SearchError

    def _convertir(self, resultats) -> list[dict[str, Any]]:
        return [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in resultats]

    def vector_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        try:
            return self._convertir(self._moteur.vector_search(query, top_k=top_k))
        except self._SearchErrorMembre2 as erreur:
            raise SearchError(str(erreur)) from erreur

    def keyword_search(self, query: str, top_k: int = 5, *, fuzzy: bool = False) -> list[dict[str, Any]]:
        try:
            return self._convertir(self._moteur.keyword_search(query, top_k=top_k, fuzzy=fuzzy))
        except self._SearchErrorMembre2 as erreur:
            raise SearchError(str(erreur)) from erreur

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        try:
            return self._convertir(self._moteur.hybrid_search(query, top_k=top_k))
        except self._SearchErrorMembre2 as erreur:
            raise SearchError(str(erreur)) from erreur


def get_search_adapter(backend: str | None = None) -> SearchAdapter:
    """Fabrique : choisit le backend à partir de l'argument, ou à défaut de la
    variable d'environnement SEARCH_BACKEND (config/settings.py), avec repli
    automatique sur le mock si le backend réel demandé n'est pas disponible
    (dépendances manquantes, Oracle non démarré, etc.) — pour que l'agent
    reste démontrable en toutes circonstances.
    """
    from config.settings import SEARCH_BACKEND

    backend = (backend or SEARCH_BACKEND or "mock").lower()
    if backend == "mock":
        return MockSearchAdapter()
    if backend == "real":
        try:
            return RealSearchAdapter()
        except SearchError as erreur:
            logger.warning(
                "Backend 'real' demandé mais indisponible (%s). "
                "Repli automatique sur MockSearchAdapter.",
                erreur,
            )
            return MockSearchAdapter()
    raise SearchError(f"Backend de recherche inconnu : '{backend}'. Utilisez 'mock' ou 'real'.")
