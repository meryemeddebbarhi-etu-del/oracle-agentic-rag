"""Construction du graphe LangGraph (principe ReAct) utilisé en mode
"auto" : c'est ici que le LLM raisonne (Reason) puis choisit et appelle un
outil de recherche (Act), observe le résultat, et recommence si besoin,
jusqu'à produire une réponse finale.

On s'appuie sur `langgraph.prebuilt.create_react_agent`, qui implémente
exactement ce cycle Reason→Act→Observe sous forme d'un graphe à deux nœuds
("agent" qui appelle le LLM, "tools" qui exécute les outils demandés), avec
une boucle conditionnelle : tant que le LLM demande un outil, on repasse par
"tools" ; dès qu'il répond sans appel d'outil, le graphe se termine. C'est le
même principe que l'agent du notebook fourni par le professeur (section 10),
étendu ici à 3 outils de recherche au lieu d'un seul, pour que l'agent
choisisse lui-même la méthode (point 2 du cahier des charges).
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from agent_search import SearchAdapter

from .prompts import SYSTEM_PROMPT_REACT
from .tools import ToolCallRecorder, create_search_tools


def build_agent_executor(llm, adapter: SearchAdapter, top_k: int = 5):
    """Retourne (graphe_compile, recorder).

    `recorder` doit être réinitialisé (`recorder.reset()`) avant chaque
    nouvelle question, puis relu après `graphe_compile.invoke(...)` pour
    récupérer les résultats bruts de recherche (voir agent.py) : c'est plus
    fiable que de re-parser le texte que le LLM a produit.
    """
    recorder = ToolCallRecorder()
    tools = create_search_tools(adapter, recorder, top_k=top_k)
    graphe = create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT_REACT)
    return graphe, recorder

