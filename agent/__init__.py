"""Agent IA et génération des réponses (Membre 3).

    from agent import RAGAgent

    agent = RAGAgent()                      # backend mock, LLM selon .env
    reponse = agent.ask("Comment fonctionne DHCP ?", method="auto")
    print(reponse["answer"])
"""

from .agent import RAGAgent

__all__ = ["RAGAgent"]
