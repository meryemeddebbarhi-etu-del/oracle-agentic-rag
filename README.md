# Oracle Agentic RAG Hybrid Search

Ce dépôt réunit l'ingestion Oracle (membre 1), les recherches vectorielle, mot-clé et hybride (membre 2), puis un agent LangGraph ReAct qui construit les réponses avec leurs sources (membre 3). L'interface Streamlit pourra appeler `RAGAgent.ask()` et afficher son résultat structuré.

## Installation

Avec Python 3.11 ou plus, dans un terminal ouvert à la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si `.env` n'existe pas encore, crée-le avec `Copy-Item .env.example .env`. S'il existe déjà, garde-le et ajoute seulement les paramètres LLM manquants ci-dessous. Complète `.env` avec vos propres identifiants Oracle et la clé du fournisseur LLM choisi. Ne partage pas et ne publie jamais `.env`.

Le ZIP fusionné ne contient pas `.env`; il ne remplace donc pas tes secrets locaux. Si ton dépôt contient encore l'ancien fichier simple `agent.py` créé avant cette fusion, supprime-le avant extraction : l'agent intégré est le package `agent/`.

## LLM et clé API

Le code utilise Groq par défaut. Crée une clé dans [GroqCloud API Keys](https://console.groq.com/keys), puis renseigne dans `.env` :

```env
LLM_PROVIDER=groq
GROQ_API_KEY=ta_cle_groq
GROQ_MODEL=openai/gpt-oss-120b
```

`openai/gpt-oss-120b` est un nom de modèle hébergé par Groq; la clé reste une clé Groq. Le modèle configuré auparavant (`llama-3.3-70b-versatile`) a été déprécié pour les comptes Free et Developer en août 2026. Vérifie les limites et le tarif du palier affiché dans ton compte avant d'utiliser l'API.

Sans API distante, tu peux mettre `LLM_PROVIDER=fake` pour une démonstration hors ligne avec le backend mock et une méthode directe (`hybrid`, `vector` ou `keyword`). Le faux modèle renvoie un texte de démonstration, pas une vraie réponse rédigée. Le mode `auto` utilise l'appel d'outils LangGraph et demande un LLM compatible avec le tool calling.

L'alternative OpenAI est également prévue dans le code :

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=ta_cle_openai
OPENAI_MODEL=gpt-4o-mini
```

L'API OpenAI se facture séparément d'un abonnement ChatGPT.

## Organisation de la recherche et de l'agent

- `search/` contient le vrai moteur du membre 2 : `vector_search`, `keyword_search`, `hybrid_search` et `SearchResult`.
- `agent_search/` contient le pont de l'agent, le mode mock et `RealSearchAdapter`. En backend `real`, ce pont importe le package Oracle du membre 2 sous le nom `search` et convertit ses `SearchResult` via `to_dict()`.
- `agent/graph.py` construit le graphe ReAct avec trois outils de recherche. Le modèle choisit l'outil en mode `auto`.
- `agent/agent.py` est le point d'entrée `RAGAgent.ask(...)`; la réponse est un dictionnaire avec la réponse, les sources, la méthode choisie, les résultats et les éventuelles erreurs.
- `agent/context_builder.py` et `agent/response_generator.py` préparent le contexte, les citations et le cas où les documents ne suffisent pas.
- `config/settings.py` lit les variables de configuration depuis `.env`.

Le conflit de nom entre les deux dossiers `search/` est résolu en conservant `search/` pour le moteur Oracle du membre 2 et en renommant le pont mock de l'agent `agent_search/`.

## Étapes de lancement

### 1. Démo hors ligne, sans Oracle ni clé LLM

Dans `.env`, choisis `SEARCH_BACKEND=mock`, `LLM_PROVIDER=fake`, `DEFAULT_SEARCH_METHOD=hybrid`, puis :

```powershell
python main.py --backend mock --method hybrid --question "Comment fonctionne DHCP ?"
```

Cette étape montre le passage de la question par le moteur mock, la structure de la réponse et les sources simulées. Le faux modèle ne rédige pas une vraie explication.

### 2. Agent LangGraph avec Groq et recherche mock

Après avoir ajouté une clé Groq dans `.env` :

```powershell
python main.py --backend mock --method auto --question "Comment fonctionne DHCP ?"
```

En mode `auto`, le graphe ReAct décide d'appeler la recherche vectorielle, mot-clé ou hybride, puis rédige une réponse fondée sur les passages trouvés.

### 3. Recherche réelle dans Oracle

Avant de lancer l'agent en mode réel, la base Oracle doit être accessible et remplie par le module d'ingestion. Vérifie avec le membre 1 avant d'exécuter `database/ingestion.py` : **ce script supprime et recrée la table `DOCUMENTS_RAG` à chaque exécution**.

Quand la table et ses index sont prêts, mets `SEARCH_BACKEND=real` dans `.env`, puis lance :

```powershell
python main.py --backend real --method hybrid --question "Comment fonctionne DHCP ?"
python main.py --backend real --method auto --question "Comment fonctionne DHCP ?"
```

Le premier appel laisse la méthode hybride répondre directement; le deuxième laisse LangGraph choisir et appeler ses outils. Les deux ont besoin d'un LLM configuré pour une réponse rédigée.

## Interface Streamlit (membre 4)

L'application peut créer une ressource `RAGAgent` une seule fois, puis faire :

```python
from agent import RAGAgent

agent = RAGAgent()
resultat = agent.ask(question_utilisateur, method=methode_choisie)
st.write(resultat["answer"])
for source in resultat["sources"]:
    st.caption(source["source"])
if resultat["error"]:
    st.error(resultat["error"])
```

Les méthodes acceptées sont `auto`, `vector`, `keyword` et `hybrid`. Le détail du résultat inclut aussi `no_answer_found`, `ambiguous` et `raw_results`.
