"""Tests automatisés de l'agent IA (Membre 3).

Ces tests sont HERMÉTIQUES : aucune connexion Oracle, aucune clé API, aucun
accès réseau. Ils utilisent un adapter de recherche factice (FakeAdapter), un
LLM factice (FakeChatLLM / LLM qui lève une erreur) et, pour le mode "auto",
un faux graphe LangGraph.

Ils vérifient surtout le contrat avec Streamlit (Membre 4) :
  - ask() ne lève JAMAIS d'exception ;
  - le JSON renvoyé a toujours les mêmes clés ;
  - chaque erreur (question vide, méthode inconnue, moteur de recherche en
    panne, clé API invalide, erreur inattendue) donne un message propre dans
    le champ "error".

Lancer :  pytest tests/test_agent.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import RAGAgent
from agent.context_builder import (
    build_context,
    collect_sources,
    detect_ambiguous,
    detect_no_answer,
)
from agent.llm_config import FakeChatLLM, LLMConfigError
from agent.prompts import AMBIGUOUS_PREFIX, NO_ANSWER_MESSAGE
from agent.tools import ToolCallRecorder
from agent_search import MockSearchAdapter, SearchAdapter, SearchError, get_search_adapter

CLES_PAYLOAD = {
    "question",
    "answer",
    "sources",
    "search_method",
    "no_answer_found",
    "ambiguous",
    "error",
    "raw_results",
}

# Score très élevé : dépasse les seuils de TOUTES les méthodes, quelle que soit
# la valeur de SEUIL_NO_ANSWER_* dans le .env de la personne qui lance les tests.
SCORE_ELEVE = 1000.0


# ---------------------------------------------------------------------------
# Outils de test
# ---------------------------------------------------------------------------


def _res(chunk_id, document_id, score, method="vector", content="contenu", metadata=None):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "content": content,
        "source": document_id,
        "score": score,
        "search_method": method,
        "metadata": metadata or {},
    }


class FakeAdapter(SearchAdapter):
    """Adapter contrôlé par le test : renvoie des résultats fixes, ou lève une erreur."""

    def __init__(self, resultats=None, erreur=None):
        self.resultats = resultats if resultats is not None else []
        self.erreur = erreur
        self.appels = []

    def _chercher(self, methode, query, top_k):
        self.appels.append((methode, query, top_k))
        if self.erreur:
            raise self.erreur
        return [dict(r, search_method=methode) for r in self.resultats]

    def vector_search(self, query, top_k=5):
        return self._chercher("vector", query, top_k)

    def keyword_search(self, query, top_k=5, *, fuzzy=False):
        return self._chercher("keyword", query, top_k)

    def hybrid_search(self, query, top_k=5):
        return self._chercher("hybrid", query, top_k)


class LLMQuiLeve:
    """LLM dont .invoke() lève toujours l'exception donnée (et compte les appels)."""

    def __init__(self, erreur):
        self.erreur = erreur
        self.nb_appels = 0

    def invoke(self, _messages):
        self.nb_appels += 1
        raise self.erreur


class FauxGraphe:
    """Simule le graphe LangGraph du mode "auto".

    `comportements` : liste consommée appel après appel. Une exception est
    levée ; une chaîne est renvoyée comme réponse finale du LLM.
    Si `resultats` est fourni, ils sont enregistrés dans le recorder (comme
    le ferait un vrai appel d'outil de recherche).
    """

    def __init__(self, recorder, comportements, resultats=None, methode="vector"):
        self.recorder = recorder
        self.comportements = list(comportements)
        self.resultats = resultats
        self.methode = methode
        self.nb_appels = 0

    def invoke(self, _entree):
        self.nb_appels += 1
        comportement = self.comportements.pop(0)
        if isinstance(comportement, Exception):
            raise comportement
        if self.resultats is not None:
            self.recorder.enregistrer(self.methode, "requete", self.resultats)
        return {"messages": [SimpleNamespace(content=comportement)]}


def _agent_auto(comportements, resultats=None, methode="vector", adapter=None):
    """Construit un RAGAgent dont le graphe LangGraph est remplacé par un FauxGraphe."""
    agent = RAGAgent(adapter=adapter or FakeAdapter(resultats=[_res("c1", "doc_a.txt", SCORE_ELEVE)]),
                     llm=FakeChatLLM("Réponse factice [source : doc_a.txt]"))
    recorder = ToolCallRecorder()
    agent._recorder = recorder
    agent._agent_graph = FauxGraphe(recorder, comportements, resultats=resultats, methode=methode)
    return agent


def _agent_direct(resultats=None, erreur=None, llm=None):
    adapter = FakeAdapter(resultats=resultats, erreur=erreur)
    return RAGAgent(adapter=adapter, llm=llm or FakeChatLLM("Réponse factice [source : doc_a.txt]"))


def _verifier_schema(payload):
    assert set(payload.keys()) == CLES_PAYLOAD


# ---------------------------------------------------------------------------
# 1. Validation des entrées
# ---------------------------------------------------------------------------


def test_question_vide_renvoie_une_erreur_propre():
    for question in ("", "   ", None):
        payload = _agent_direct().ask(question)
        _verifier_schema(payload)
        assert "vide" in payload["error"].lower()
        assert payload["answer"] == ""
        assert payload["no_answer_found"] is True


def test_methode_inconnue_renvoie_une_erreur_propre():
    payload = _agent_direct().ask("Qu'est-ce qu'un rôle ?", method="magique")
    _verifier_schema(payload)
    assert "Méthode inconnue" in payload["error"]
    assert "magique" in payload["error"]


def test_la_methode_est_insensible_a_la_casse():
    payload = _agent_direct([_res("c1", "doc_a.txt", SCORE_ELEVE)]).ask("Question ?", method="HYBRID")
    assert payload["error"] is None
    assert payload["search_method"] == "hybrid"


# ---------------------------------------------------------------------------
# 2. Chemin direct (vector / keyword / hybrid)
# ---------------------------------------------------------------------------


def test_chemin_direct_succes_pour_chaque_methode():
    for methode in ("vector", "keyword", "hybrid"):
        agent = _agent_direct([_res("c1", "doc_a.txt", SCORE_ELEVE)])
        payload = agent.ask("Quels sont les modes SHUTDOWN ?", method=methode, top_k=3)
        _verifier_schema(payload)
        assert payload["error"] is None
        assert payload["search_method"] == methode
        assert payload["no_answer_found"] is False
        assert "Réponse factice" in payload["answer"]
        assert payload["raw_results"]
        # la méthode et top_k demandés sont bien transmis à l'adapter
        assert agent.adapter.appels == [(methode, "Quels sont les modes SHUTDOWN ?", 3)]


def test_les_sources_citees_dans_la_reponse_sont_seules_gardees():
    resultats = [_res("c1", "doc_a.txt", SCORE_ELEVE), _res("c2", "doc_b.txt", SCORE_ELEVE - 1)]
    # le LLM factice ne cite que doc_a.txt
    payload = _agent_direct(resultats).ask("Question ?", method="vector")
    assert [s["document_id"] for s in payload["sources"]] == ["doc_a.txt"]


def test_absence_de_resultats_donne_le_message_standard_sans_appeler_le_llm():
    llm = LLMQuiLeve(RuntimeError("ne doit pas être appelé"))
    payload = _agent_direct(resultats=[], llm=llm).ask("Recette du couscous ?", method="hybrid")
    _verifier_schema(payload)
    assert payload["error"] is None
    assert payload["no_answer_found"] is True
    assert payload["answer"] == NO_ANSWER_MESSAGE
    assert payload["sources"] == []
    assert payload["ambiguous"] is False
    assert llm.nb_appels == 0


def test_resultats_sous_le_seuil_comptent_comme_aucune_reponse():
    # score 0.0 : sous le seuil de toutes les méthodes (vector, keyword, hybrid)
    payload = _agent_direct([_res("c1", "doc_a.txt", 0.0)]).ask("Question hors sujet ?", method="vector")
    assert payload["no_answer_found"] is True
    assert payload["answer"] == NO_ANSWER_MESSAGE


def test_question_ambigue_ajoute_le_prefixe_et_le_drapeau():
    resultats = [
        _res("c1", "doc_a.txt", SCORE_ELEVE),
        _res("c2", "doc_b.txt", SCORE_ELEVE - 50),  # écart < 15 % d'un autre document
    ]
    payload = _agent_direct(resultats).ask("Qu'est-ce qu'un rôle ?", method="vector")
    assert payload["ambiguous"] is True
    assert payload["answer"].startswith(AMBIGUOUS_PREFIX)


def test_pas_d_ambiguite_si_meme_document_ou_methode_hybrid():
    meme_doc = [_res("c1", "doc_a.txt", SCORE_ELEVE), _res("c2", "doc_a.txt", SCORE_ELEVE - 1)]
    assert _agent_direct(meme_doc).ask("Q ?", method="vector")["ambiguous"] is False
    deux_docs = [_res("c1", "doc_a.txt", SCORE_ELEVE), _res("c2", "doc_b.txt", SCORE_ELEVE - 1)]
    assert _agent_direct(deux_docs).ask("Q ?", method="hybrid")["ambiguous"] is False


# ---------------------------------------------------------------------------
# 3. Gestion des erreurs (point 10 du cahier des charges)
# ---------------------------------------------------------------------------


def test_moteur_de_recherche_en_panne_renvoie_une_erreur_propre():
    erreur = SearchError("Connexion à Oracle impossible : DPY-6005")
    for methode in ("vector", "keyword", "hybrid"):
        payload = _agent_direct(erreur=erreur).ask("Quels sont les modes SHUTDOWN ?", method=methode)
        _verifier_schema(payload)
        assert payload["error"].startswith("Erreur du moteur de recherche")
        assert "Connexion à Oracle impossible" in payload["error"]
        assert payload["answer"] == ""
        assert payload["no_answer_found"] is True
        assert payload["raw_results"] == []


def test_cle_api_invalide_renvoie_un_message_clair():
    llm = LLMQuiLeve(Exception("Error code: 401 - {'error': {'message': 'Invalid API Key', 'code': 'invalid_api_key'}}"))
    payload = _agent_direct([_res("c1", "doc_a.txt", SCORE_ELEVE)], llm=llm).ask("Question ?", method="hybrid")
    _verifier_schema(payload)
    assert "Clé API du LLM invalide" in payload["error"]
    assert "GROQ_API_KEY" in payload["error"]
    assert "Traceback" not in payload["error"]


def test_erreur_de_configuration_du_llm():
    llm = LLMQuiLeve(LLMConfigError("GROQ_API_KEY est absent de .env"))
    payload = _agent_direct([_res("c1", "doc_a.txt", SCORE_ELEVE)], llm=llm).ask("Question ?", method="hybrid")
    assert payload["error"].startswith("Erreur de configuration du LLM")
    assert "GROQ_API_KEY" in payload["error"]


def test_erreur_inattendue_ne_fait_jamais_planter_ask():
    llm = LLMQuiLeve(ValueError("boum"))
    payload = _agent_direct([_res("c1", "doc_a.txt", SCORE_ELEVE)], llm=llm).ask("Question ?", method="hybrid")
    _verifier_schema(payload)
    assert payload["error"].startswith("Erreur inattendue")
    assert "boum" in payload["error"]


# ---------------------------------------------------------------------------
# 4. Mode "auto" (agent ReAct simulé)
# ---------------------------------------------------------------------------


def test_auto_succes_utilise_les_methodes_enregistrees():
    resultats = [_res("c1", "doc_a.txt", SCORE_ELEVE)]
    agent = _agent_auto(["Un rôle regroupe des privilèges. [source : doc_a.txt]"], resultats=resultats, methode="keyword")
    payload = agent.ask("Qu'est-ce qu'un rôle ?", method="auto")
    _verifier_schema(payload)
    assert payload["error"] is None
    assert payload["search_method"] == "keyword"
    assert payload["answer"].startswith("Un rôle regroupe")
    assert payload["no_answer_found"] is False
    assert [s["document_id"] for s in payload["sources"]] == ["doc_a.txt"]


def test_auto_plusieurs_methodes_sont_jointes_par_un_plus():
    agent = _agent_auto(["Réponse [source : doc_a.txt]"], resultats=[_res("c1", "doc_a.txt", SCORE_ELEVE)])
    agent._agent_graph.invoke = lambda _e: _deux_recherches(agent)
    payload = agent.ask("Question ?", method="auto")
    assert payload["search_method"] == "vector+keyword"


def _deux_recherches(agent):
    agent._recorder.enregistrer("vector", "q", [_res("c1", "doc_a.txt", SCORE_ELEVE)])
    agent._recorder.enregistrer("keyword", "q", [_res("c1", "doc_a.txt", SCORE_ELEVE)])
    return {"messages": [SimpleNamespace(content="Réponse [source : doc_a.txt]")]}


def test_auto_sans_aucun_appel_d_outil_donne_aucune_reponse():
    agent = _agent_auto(["Je réponds de mémoire."], resultats=None)
    payload = agent.ask("Question ?", method="auto")
    assert payload["search_method"] == "aucune"
    assert payload["no_answer_found"] is True
    assert payload["sources"] == []


def test_auto_le_llm_peut_declarer_lui_meme_l_absence_de_reponse():
    agent = _agent_auto(
        ["Je ne trouve pas cette information dans les documents disponibles."],
        resultats=[_res("c1", "doc_a.txt", SCORE_ELEVE)],
    )
    payload = agent.ask("Question ?", method="auto")
    assert payload["no_answer_found"] is True
    assert payload["sources"] == []


def test_auto_reessaie_apres_un_appel_d_outil_invalide():
    erreur_outil = Exception("Error code: 400 - tool_use_failed: tool call validation failed")
    agent = _agent_auto([erreur_outil, "Réponse finale [source : doc_a.txt]"],
                        resultats=[_res("c1", "doc_a.txt", SCORE_ELEVE)])
    payload = agent.ask("Question ?", method="auto")
    assert payload["error"] is None
    assert payload["answer"].startswith("Réponse finale")
    assert agent._agent_graph.nb_appels == 2


def test_auto_se_replie_sur_hybrid_apres_trois_echecs_d_outil():
    erreur_outil = Exception("tool_use_failed")
    agent = _agent_auto([erreur_outil, erreur_outil, erreur_outil])
    payload = agent.ask("Question ?", method="auto")
    _verifier_schema(payload)
    assert payload["error"] is None
    assert payload["search_method"] == "hybrid"
    assert agent._agent_graph.nb_appels == 3  # 1 essai + 2 retries
    assert agent.adapter.appels and agent.adapter.appels[0][0] == "hybrid"


def test_auto_cle_api_invalide_n_est_pas_reessayee():
    agent = _agent_auto([Exception("Error code: 401 - Invalid API Key")])
    payload = agent.ask("Question ?", method="auto")
    assert "Clé API du LLM invalide" in payload["error"]
    assert agent._agent_graph.nb_appels == 1


# ---------------------------------------------------------------------------
# 5. Briques internes : contexte, sources, seuils, recorder, adapter
# ---------------------------------------------------------------------------


def test_build_context_vide_et_dedoublonnage_par_chunk_id():
    assert build_context([]) == ""
    contexte = build_context([_res("c1", "doc_a.txt", 1, content="AAA"), _res("c1", "doc_a.txt", 1, content="AAA")])
    assert contexte.count("AAA") == 1
    assert "[Source : doc_a.txt]" in contexte


def test_build_context_respecte_la_taille_maximale():
    gros = [_res(f"c{i}", f"doc_{i}.txt", 1, content="x" * 500) for i in range(10)]
    contexte = build_context(gros, max_chars=1200)
    assert len(contexte) < 1700  # 2 blocs entiers + séparateurs, pas les 10
    assert contexte.count("[Source :") < 10


def test_collect_sources_garde_la_premiere_occurrence():
    sources = collect_sources([
        _res("c1", "doc_a.txt", 0.9, method="vector"),
        _res("c2", "doc_a.txt", 0.5, method="keyword"),
        _res("c3", "doc_b.txt", 0.4, method="vector"),
    ])
    assert [s["source"] for s in sources] == ["doc_a.txt", "doc_b.txt"]
    assert sources[0]["score"] == 0.9 and sources[0]["search_method"] == "vector"


def test_detect_no_answer():
    assert detect_no_answer([]) is True
    assert detect_no_answer([_res("c1", "d.txt", SCORE_ELEVE, method="keyword")]) is False
    assert detect_no_answer([_res("c1", "d.txt", 0.0, method="vector")]) is True


def test_detect_no_answer_hybrid_juge_sur_les_scores_d_origine():
    # Le score RRF est minuscule par construction : on regarde vector_score / keyword_score.
    bon = _res("c1", "d.txt", 0.0001, method="hybrid", metadata={"vector_score": 0.99})
    mauvais = _res("c1", "d.txt", 0.0001, method="hybrid", metadata={"vector_score": -0.5, "keyword_score": 0})
    assert detect_no_answer([bon]) is False
    assert detect_no_answer([mauvais]) is True


def test_detect_ambiguous_scores_eloignes():
    resultats = [_res("c1", "doc_a.txt", SCORE_ELEVE), _res("c2", "doc_b.txt", 10.0)]
    assert detect_ambiguous(resultats) is False
    assert detect_ambiguous([]) is False


def test_recorder_fusionne_par_methode_et_garde_le_meilleur_score():
    recorder = ToolCallRecorder()
    recorder.enregistrer("vector", "q1", [_res("c1", "doc_a.txt", 0.3)])
    recorder.enregistrer("vector", "q2", [_res("c1", "doc_a.txt", 0.8), _res("c2", "doc_b.txt", 0.5)])
    recorder.enregistrer("keyword", "q1", [_res("c1", "doc_a.txt", 20, method="keyword")])

    assert recorder.methodes_utilisees() == ["vector", "keyword"]
    vecteur = [r for r in recorder.tous_les_resultats() if r["search_method"] == "vector"]
    assert {r["chunk_id"]: r["score"] for r in vecteur} == {"c1": 0.8, "c2": 0.5}

    recorder.reset()
    assert recorder.tous_les_resultats() == [] and recorder.methodes_utilisees() == []


def test_adapter_generique_refuse_une_methode_inconnue():
    with pytest.raises(SearchError):
        FakeAdapter().search("question", method="inconnue")


def test_fabrique_refuse_un_backend_inconnu():
    with pytest.raises(SearchError):
        get_search_adapter("backend_qui_n_existe_pas")


def test_fabrique_mock_renvoie_le_mock():
    assert isinstance(get_search_adapter("mock"), MockSearchAdapter)


def test_agent_avec_le_vrai_mock_respecte_le_schema():
    """Test de bout en bout léger avec le corpus simulé (aucune dépendance externe)."""
    agent = RAGAgent(backend="mock", llm=FakeChatLLM("Réponse factice"))
    for methode in ("vector", "keyword", "hybrid"):
        payload = agent.ask("Comment fonctionne DHCP ?", method=methode)
        _verifier_schema(payload)
        assert payload["error"] is None
        assert payload["search_method"] == methode