"""Configuration du LLM — point UNIQUE de choix/instanciation du modèle de
langage, conformément au cahier des charges ("Configuration du modèle dans
un fichier séparé").

Priorité donnée aux solutions gratuites ou à free tier :
  - Groq (par défaut) : free tier généreux, modèles Llama 3.x avec appel
    d'outils (tool calling) compatible LangGraph/create_react_agent.
  - OpenAI : utilisable si LLM_PROVIDER=openai et OPENAI_API_KEY fourni.
  - fake   : aucune clé requise, aucun appel réseau — pour les tests
    automatiques et les démonstrations hors-ligne (voir FakeChatLLM).
"""

from __future__ import annotations

from config import settings


class LLMConfigError(Exception):
    """Levée quand le LLM demandé ne peut pas être instancié (clé manquante...)."""


class FakeChatLLM:
    """LLM factice, sans aucun appel réseau : utilisé par les tests et par le
    mode de secours (démo possible même sans clé API). Expose seulement
    `.invoke(...)`, comme un modèle LangChain minimal, pour les usages de
    agent/response_generator.py qui n'ont pas besoin d'appel d'outils.
    """

    def __init__(self, reponse: str | None = None) -> None:
        self._reponse = reponse

    class _Resultat:
        def __init__(self, content: str) -> None:
            self.content = content

    def invoke(self, messages) -> "FakeChatLLM._Resultat":
        if self._reponse is not None:
            return FakeChatLLM._Resultat(self._reponse)
        # Réponse déterministe et sans réseau : utile en démonstration si
        # aucune clé API (Groq/OpenAI) n'est configurée.
        return FakeChatLLM._Resultat(
            "[Mode démo sans LLM] Voici les passages les plus pertinents "
            "trouvés dans les documents ; configurez GROQ_API_KEY ou "
            "OPENAI_API_KEY dans .env pour obtenir une réponse rédigée."
        )


def get_llm(temperature: int | None = None):
    """Retourne une instance de modèle de chat prête à l'emploi (interface
    LangChain : `.invoke(...)`, et pour Groq/OpenAI `.bind_tools(...)` pour
    l'agent ReAct).

    Lève LLMConfigError avec un message explicite si le fournisseur demandé
    n'a pas sa clé API configurée — plutôt que de laisser remonter une
    erreur réseau opaque au premier appel.
    """
    provider = (settings.LLM_PROVIDER or "groq").lower()
    temperature = settings.LLM_TEMPERATURE if temperature is None else temperature

    if provider == "fake":
        return FakeChatLLM()

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise LLMConfigError(
                "LLM_PROVIDER=groq mais GROQ_API_KEY est absent de .env. "
                "Créez une clé gratuite sur https://console.groq.com puis "
                "ajoutez GROQ_API_KEY=... à votre fichier .env."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as erreur:
            raise LLMConfigError(
                "Le paquet 'langchain-groq' n'est pas installé "
                "(pip install langchain-groq)."
            ) from erreur
        return ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=temperature)

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise LLMConfigError(
                "LLM_PROVIDER=openai mais OPENAI_API_KEY est absent de .env."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as erreur:
            raise LLMConfigError(
                "Le paquet 'langchain-openai' n'est pas installé "
                "(pip install langchain-openai)."
            ) from erreur
        return ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY, temperature=temperature)

    raise LLMConfigError(
        f"LLM_PROVIDER='{provider}' inconnu. Valeurs acceptées : groq, openai, fake."
    )
