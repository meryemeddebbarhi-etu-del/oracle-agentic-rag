# Documentation technique — Module Agent IA (Membre 3)

Ce document détaille le rôle de chaque fichier et de chaque fonction
principale, pour préparer la soutenance.

## agent_search/errors.py

- `SearchError(Exception)` : exception unique utilisée par **tout** le
  module de recherche, qu'il s'agisse du mock ou du vrai moteur (converti
  dans `RealSearchAdapter`). Isolée dans son propre fichier pour éviter un
  import circulaire entre `search_adapter.py` et `mock_search.py`.

## agent_search/mock_search.py

- `_CORPUS` : liste de 6 passages simulés, repris des mêmes thématiques que
  les documents réels du projet (réseaux : OSI, TCP/IP, DNS, DHCP, HTTP).
- `vector_search(query, top_k)` : score = proportion de mots de la requête
  retrouvés (après normalisation accents/casse) dans chaque document. Une
  approximation volontairement simple d'une similarité sémantique.
- `keyword_search(query, top_k, fuzzy=False)` : score = nombre
  d'occurrences des mots de la requête (façon Oracle Text `SCORE()`) ; option
  `fuzzy` tolère les fautes de frappe via comparaison de préfixes.
- `hybrid_search(query, top_k, candidate_k, rrf_k, w_vector, w_keyword)` :
  fusionne les deux listes précédentes par **Reciprocal Rank Fusion (RRF)** —
  identique au principe utilisé par le vrai moteur du Membre 2 — en ne
  prenant en compte que le **rang** de chaque document dans chaque liste (pas
  son score brut, les deux échelles n'étant pas comparables).

## agent_search/search_adapter.py

- `SearchAdapter(ABC)` : interface commune (`vector_search`,
  `keyword_search`, `hybrid_search`, et `search(query, top_k, method)` qui
  redirige vers la bonne méthode).
- `MockSearchAdapter` : délègue à `mock_search.py`. Backend par défaut.
- `RealSearchAdapter` : importe paresseusement (dans `__init__`, pas en haut
  du fichier) le module `search` du Membre 2, pour que le reste du projet
  fonctionne même si `oracledb` ou Oracle ne sont pas installés/démarrés.
  Convertit chaque `SearchResult` (dataclass du Membre 2) en dict via
  `.to_dict()`, et reconvertit `search.SearchError` en `SearchError` local,
  pour que l'agent n'ait jamais besoin de connaître le type d'exception du
  Membre 2.
- `get_search_adapter(backend=None)` : fabrique. Lit `SEARCH_BACKEND` dans
  `config/settings.py` si `backend` n'est pas fourni. Si `backend="real"`
  est demandé mais échoue (dépendance manquante, Oracle éteint), **bascule
  automatiquement sur `MockSearchAdapter`** avec un avertissement journalisé
  (`logging.warning`), plutôt que de faire planter l'application.

## config/settings.py

Centralise toutes les variables d'environnement (voir `.env.example`) :
backend de recherche, fournisseur et modèle LLM, clés API, seuils de
détection d'absence de réponse (différents par méthode, car les échelles de
score de vector/keyword/hybrid ne sont pas comparables), `top_k` par défaut.

## agent/llm_config.py

- `get_llm(temperature=None)` : fabrique le LLM selon `LLM_PROVIDER`
  (`groq` par défaut, `openai`, ou `fake`). Lève `LLMConfigError` avec un
  message explicite si la clé API correspondante est absente, plutôt que de
  laisser remonter une erreur réseau cryptique au premier appel.
- `FakeChatLLM` : implémente seulement `.invoke(...)`, aucun appel réseau.
  Utilisé par les tests (`tests/conftest.py::FakeLLM`, variante encore plus
  simple avec compteur d'appels) et comme filet de sécurité pour une
  démonstration sans aucune clé API configurée.

## agent/prompts.py

Regroupe tous les textes envoyés au LLM : `SYSTEM_PROMPT_REACT` (explique au
LLM les 3 outils et les règles — toujours chercher avant de répondre, ne
jamais inventer, toujours citer les sources, signaler les questions
ambiguës), `RESPONSE_PROMPT_TEMPLATE` (mode direct, sans boucle d'outils),
`NO_ANSWER_MESSAGE` et `AMBIGUOUS_PREFIX` (textes fixes, utilisés sans appel
LLM pour les cas détectés automatiquement — déterministe, rapide, jamais
d'hallucination sur ces deux cas précis).

## agent/tools.py

- `ToolCallRecorder` : objet qui enregistre, pendant UNE invocation de
  l'agent, chaque appel d'outil effectué (méthode, requête, résultats
  bruts). `agent.py` relit cet historique après l'exécution du graphe pour
  reconstruire les sources et détecter l'absence de réponse **sans dépendre
  du texte produit par le LLM**, ce qui est plus robuste (le LLM pourrait
  paraphraser ou omettre une source dans sa réponse finale).
- `create_search_tools(adapter, recorder, top_k)` : construit dynamiquement
  les 3 `@tool` LangChain (`search_vector`, `search_keyword`,
  `search_hybrid`), chacun fermé (closure) sur un `adapter` et un `recorder`
  donnés. Chaque outil attrape `SearchError` et renvoie un message texte
  explicite au LLM (plutôt qu'une exception qui ferait planter le graphe).

## agent/context_builder.py

- `build_context(results, max_chars)` : concatène les passages en texte,
  borné en taille (3000 caractères par défaut) pour ne pas dépasser la
  fenêtre de contexte du LLM.
- `collect_sources(results)` : déduplique par `source`, garde le meilleur
  score si une même source apparaît plusieurs fois (plusieurs morceaux d'un
  même document retrouvés), trie par score décroissant.
- `detect_no_answer(results, method)` : renvoie `True` si la liste est vide,
  ou si **tous** les résultats sont sous le seuil de leur méthode
  (`config.settings.SEUIL_VECTOR/KEYWORD/HYBRID`).
- `detect_ambiguous(results)` : renvoie `True` si au moins deux **documents
  différents** (pas deux morceaux du même document) ont un score à moins de
  15 % du meilleur score — signe que plusieurs réponses concurrentes
  existent dans le corpus.

## agent/response_generator.py

- `generate_answer(llm, question, results, no_answer, ambiguous)` : si
  `no_answer` est vrai, renvoie directement `NO_ANSWER_MESSAGE` **sans
  appeler le LLM** (testé explicitement dans
  `tests/test_responses.py::test_generate_answer_no_answer_ne_contacte_pas_le_llm`).
  Sinon, construit le prompt (`RESPONSE_PROMPT_TEMPLATE`) avec le contexte
  et appelle `llm.invoke(...)`. Préfixe la réponse si `ambiguous`.
- `build_response_payload(...)` : construit le dict JSON **au schéma
  toujours identique** (mêmes clés, y compris en cas d'erreur), consommé par
  Streamlit (Membre 4) : `question`, `answer`, `sources`, `search_method`,
  `no_answer_found`, `ambiguous`, `error`, `raw_results`.

## agent/graph.py

- `build_agent_executor(llm, adapter, top_k)` : construit le graphe
  LangGraph via `langgraph.prebuilt.create_react_agent(llm, tools=...,
  prompt=SYSTEM_PROMPT_REACT)`. Renvoie `(graphe, recorder)` : le `recorder`
  doit être réinitialisé avant chaque nouvelle question (fait dans
  `agent.py::_ask_auto`).

## agent/agent.py

- `RAGAgent.__init__(backend, llm, adapter, top_k)` : construit (ou reçoit)
  l'adapter de recherche ; le LLM est construit **paresseusement** (au
  premier accès à `self.llm`), pour ne jamais exiger de clé API si l'appelant
  fournit son propre LLM (tests).
- `RAGAgent.ask(question, method, top_k)` : point d'entrée unique.
  - Valide la question et la méthode (`vector|keyword|hybrid|auto`).
  - `method="auto"` → `_ask_auto` : construit/relit le graphe LangGraph,
    l'invoque, relit `ToolCallRecorder` pour les résultats bruts et la
    méthode effectivement choisie par le LLM.
  - `method` explicite → `_ask_direct` : appelle directement
    `adapter.search(question, top_k, method)`, construit le contexte, génère
    la réponse.
  - **Ne lève jamais d'exception** : `SearchError`, `LLMConfigError`, et
    toute autre exception sont attrapées et renvoyées dans le champ
    `"error"` du payload JSON, avec `no_answer_found=True` par sécurité.

## main.py

CLI avec `argparse` : mode interactif (boucle `input()`) ou question unique
(`--question`), choix du backend/méthode/top_k, sortie texte formatée ou
JSON brut (`--json`). Compatible Windows (aucune commande shell Unix).

## Tests

Les archives des membres contiennent les suites de tests d'origine; elles ne
sont pas recopiées dans ce paquet d'intégration. Le présent document décrit
les modules d'agent et leur rôle pour la soutenance.
