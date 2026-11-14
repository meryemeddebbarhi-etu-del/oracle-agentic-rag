"""Point d'entrée principal du module : la classe RAGAgent.

    from agent import RAGAgent

    agent = RAGAgent()                                   # mock + LLM de .env
    reponse = agent.ask("Comment fonctionne DHCP ?")      # method="auto" par défaut
    reponse = agent.ask("Qu'est-ce que DNS ?", method="keyword")

`ask()` ne lève jamais d'exception : toute erreur (moteur de recherche
indisponible, LLM mal configuré, méthode inconnue...) est attrapée et
renvoyée dans le champ "error" du payload JSON standardisé, pour que
Streamlit (Membre 4) puisse toujours afficher quelque chose de propre
(point 10 du cahier des charges : gérer les erreurs).

Deux modes :
  - method="vector"|"keyword"|"hybrid" : l'agent appelle directement cette
    méthode de recherche puis demande au LLM de rédiger la réponse. Chemin
    déterministe, ne nécessite qu'un LLM de complétion simple (fonctionne
    même avec FakeChatLLM, donc testable sans clé API).
  - method="auto" (par défaut) : un agent LangGraph ReAct (voir graph.py)
    laisse le LLM choisir lui-même, via appel d'outils, quelle(s) méthode(s)
    de recherche utiliser. Nécessite un LLM capable d'appel d'outils (Groq
    ou OpenAI, pas FakeChatLLM).
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings
from agent_search import SearchAdapter, SearchError, get_search_adapter

from .context_builder import detect_ambiguous, detect_no_answer
from .graph import build_agent_executor
from .llm_config import LLMConfigError, get_llm
from .response_generator import build_response_payload, generate_answer

logger = logging.getLogger(__name__)

_METHODES_VALIDES = {"vector", "keyword", "hybrid", "auto"}


class RAGAgent:
    def __init__(
        self,
        backend: str | None = None,
        llm: Any | None = None,
        adapter: SearchAdapter | None = None,
        top_k: int | None = None,
    ) -> None:
        """
        Args:
            backend: "mock" ou "real" (défaut : config.settings.SEARCH_BACKEND).
                     Ignoré si `adapter` est fourni directement.
            llm: instance de modèle de chat déjà construite (utile pour les
                 tests, avec un FakeChatLLM). Sinon, construite via
                 agent.llm_config.get_llm() selon .env.
            adapter: SearchAdapter déjà construit (prioritaire sur `backend`),
                     utile pour injecter un adapter de test personnalisé.
            top_k: nombre de résultats demandés par appel de recherche.
        """
        self.adapter = adapter or get_search_adapter(backend)
        self.top_k = top_k or settings.DEFAULT_TOP_K
        self._llm = llm  # construit paresseusement si None (voir self.llm)
        self._agent_graph = None
        self._recorder = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _assurer_graphe(self):
        if self._agent_graph is None:
            self._agent_graph, self._recorder = build_agent_executor(self.llm, self.adapter, top_k=self.top_k)
        return self._agent_graph, self._recorder

    def ask(self, question: str, method: str | None = None, top_k: int | None = None) -> dict[str, Any]:
        method = (method or settings.DEFAULT_SEARCH_METHOD or "auto").lower()
        top_k = top_k or self.top_k

        question = (question or "").strip()
        if not question:
            return build_response_payload(
                question=question,
                answer="",
                results=[],
                search_method=method,
                error="La question ne peut pas être vide.",
                no_answer=True,
                ambiguous=False,
            )

        if method not in _METHODES_VALIDES:
            return build_response_payload(
                question=question,
                answer="",
                results=[],
                search_method=method,
                error=f"Méthode inconnue : '{method}'. Valeurs acceptées : vector, keyword, hybrid, auto.",
                no_answer=True,
                ambiguous=False,
            )

        try:
            if method == "auto":
                return self._ask_auto(question, top_k)
            return self._ask_direct(question, method, top_k)
        except SearchError as erreur:
            logger.warning("Erreur du moteur de recherche : %s", erreur)
            return build_response_payload(
                question=question, answer="", results=[], search_method=method,
                error=f"Erreur du moteur de recherche : {erreur}", no_answer=True, ambiguous=False,
            )
        except LLMConfigError as erreur:
            logger.warning("Erreur de configuration LLM : %s", erreur)
            return build_response_payload(
                question=question, answer="", results=[], search_method=method,
                error=f"Erreur de configuration du LLM : {erreur}", no_answer=True, ambiguous=False,
            )
        except Exception as erreur:  # garde-fou final : jamais de crash pour Streamlit
            logger.exception("Erreur inattendue dans RAGAgent.ask")
            return build_response_payload(
                question=question, answer="", results=[], search_method=method,
                error=f"Erreur inattendue : {erreur}", no_answer=True, ambiguous=False,
            )

    def _ask_direct(self, question: str, method: str, top_k: int) -> dict[str, Any]:
        resultats = self.adapter.search(question, top_k=top_k, method=method)
        no_answer = detect_no_answer(resultats, method=method)
        ambiguous = detect_ambiguous(resultats)
        reponse = generate_answer(self.llm, question, resultats, no_answer=no_answer, ambiguous=ambiguous)
        return build_response_payload(
            question=question, answer=reponse, results=resultats, search_method=method,
            error=None, no_answer=no_answer, ambiguous=ambiguous,
        )

    def _ask_auto(self, question: str, top_k: int) -> dict[str, Any]:
        graphe, recorder = self._assurer_graphe()
        recorder.reset()

        sortie = graphe.invoke({"messages": [{"role": "user", "content": question}]})
        messages = sortie.get("messages", [])
        reponse_llm = messages[-1].content if messages else ""

        resultats = recorder.tous_les_resultats()
        methodes = recorder.methodes_utilisees()
        search_method = "+".join(methodes) if methodes else "aucune"

        no_answer = detect_no_answer(resultats) if resultats else True
        # Le LLM peut aussi exprimer lui-même l'absence de réponse en texte.
        if any(m in reponse_llm.lower() for m in ["ne trouve pas", "aucune information", "pas d'information"]):
            no_answer = True
        ambiguous = detect_ambiguous(resultats)

        reponse_finale = reponse_llm if reponse_llm else generate_answer(
            self.llm, question, resultats, no_answer=no_answer, ambiguous=ambiguous
        )
        return build_response_payload(
            question=question, answer=reponse_finale, results=resultats, search_method=search_method,
            error=None, no_answer=no_answer, ambiguous=ambiguous,
        )

