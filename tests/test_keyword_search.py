"""Tests d'INTÉGRATION de keyword_search() (Membre 2).

Nécessitent une vraie base Oracle avec l'index Oracle Text DOCUMENTS_RAG_TEXT_IDX
(créé par database/ingestion.py) et le corpus documents/*.txt (Oracle / PL/SQL).
Sans base joignable, la fixture `oracle_connection` SKIP le test.

Les termes de recherche ont été choisis parce qu'ils n'apparaissent que dans UN
document du corpus (vérifié avec grep), pour pouvoir affirmer quel document doit
remonter.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from search.keyword_search import keyword_search
from search.search_utils import SearchError

DUREE_MAX_S = float(os.getenv("ORACLE_TEST_MAX_SECONDS", "10"))


def _documents(resultats):
    return {r.document_id for r in resultats}


@pytest.mark.parametrize(
    "terme, document_attendu",
    [
        ("EXECUTE IMMEDIATE", "plsql_sql_dynamique_metadata.txt"),
        ("WHERE CURRENT OF", "plsql_curseurs.txt"),
        ("PACKAGE BODY", "plsql_packages.txt"),
        ("CREATE PROFILE", "oracle_utilisateurs_privileges_roles_profils.txt"),
        ("expdp", "oracle_datapump_sqlloader.txt"),
        ("tnsnames", "oracle_net_listener.txt"),
        ("SHUTDOWN", "oracle_demarrage_arret_parametres.txt"),
    ],
)
def test_keyword_search_trouve_le_document_qui_contient_le_terme(oracle_connection, terme, document_attendu):
    resultats = keyword_search(terme, top_k=10, connection=oracle_connection)
    assert document_attendu in _documents(resultats), f"'{document_attendu}' absent pour '{terme}'"


def test_keyword_search_terme_rare_ne_renvoie_que_son_document(oracle_connection):
    resultats = keyword_search("tnsnames", top_k=10, connection=oracle_connection)
    assert resultats
    assert _documents(resultats) == {"oracle_net_listener.txt"}
    assert all("tnsnames" in r.content.lower() for r in resultats)


def test_keyword_search_format_des_resultats(oracle_connection):
    resultats = keyword_search("SHUTDOWN", top_k=3, connection=oracle_connection)
    assert 1 <= len(resultats) <= 3
    for r in resultats:
        assert r.search_method == "keyword"
        assert r.chunk_id and r.content.strip()
        assert r.document_id.endswith(".txt")
        assert 0 < r.score <= 100  # score Oracle Text
        assert r.metadata["oracle_text_score"] == r.score


def test_keyword_search_scores_decroissants_et_sans_doublon(oracle_connection):
    resultats = keyword_search("PDB tablespace utilisateur", top_k=8, connection=oracle_connection)
    scores = [r.score for r in resultats]
    assert scores == sorted(scores, reverse=True)
    ids = [r.chunk_id for r in resultats]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("k", [1, 3])
def test_keyword_search_respecte_top_k(oracle_connection, k):
    assert len(keyword_search("Oracle", top_k=k, connection=oracle_connection)) == k


def test_keyword_search_est_insensible_a_la_casse(oracle_connection):
    minuscule = keyword_search("shutdown", top_k=10, connection=oracle_connection)
    majuscule = keyword_search("SHUTDOWN", top_k=10, connection=oracle_connection)
    assert {r.chunk_id for r in minuscule} == {r.chunk_id for r in majuscule}


def test_keyword_search_terme_absent_renvoie_liste_vide(oracle_connection):
    assert keyword_search("xylophoneinexistant999", top_k=5, connection=oracle_connection) == []


def test_keyword_search_fuzzy_tolere_une_faute_de_frappe(oracle_connection):
    # "SHUTDOWM" n'existe nulle part : la recherche exacte ne trouve rien,
    # la recherche floue retrouve SHUTDOWN.
    assert keyword_search("SHUTDOWM", top_k=5, connection=oracle_connection) == []
    floue = keyword_search("SHUTDOWM", top_k=5, fuzzy=True, connection=oracle_connection)
    assert "oracle_demarrage_arret_parametres.txt" in _documents(floue)


@pytest.mark.parametrize(
    "requete",
    ["qu'est-ce que le listener ?", 'SHUTDOWN "IMMEDIATE"', "ORA-01034 (instance arrêtée)", "a;b' OR '1'='1"],
)
def test_keyword_search_caracteres_speciaux_ne_cassent_pas_oracle_text(oracle_connection, requete):
    assert isinstance(keyword_search(requete, top_k=3, connection=oracle_connection), list)


def test_keyword_search_ponctuation_seule_renvoie_liste_vide(oracle_connection):
    assert keyword_search("?!...", top_k=3, connection=oracle_connection) == []


@pytest.mark.parametrize("requete", ["", "   ", None])
def test_keyword_search_requete_vide_leve_searcherror(oracle_connection, requete):
    with pytest.raises(SearchError):
        keyword_search(requete, top_k=3, connection=oracle_connection)


@pytest.mark.parametrize("top_k", [0, -1, 201, "5", True])
def test_keyword_search_top_k_invalide_leve_searcherror(oracle_connection, top_k):
    with pytest.raises(SearchError):
        keyword_search("SHUTDOWN", top_k=top_k, connection=oracle_connection)


def test_keyword_search_est_assez_rapide(oracle_connection):
    debut = time.perf_counter()
    keyword_search("SHUTDOWN IMMEDIATE", top_k=5, connection=oracle_connection)
    duree = time.perf_counter() - debut
    print(f"\nkeyword_search (k=5) : {duree * 1000:.1f} ms")
    assert duree < DUREE_MAX_S, f"trop lent : {duree:.1f} s (seuil {DUREE_MAX_S} s)"


def test_keyword_search_base_injoignable_leve_searcherror(monkeypatch):
    """N'a PAS besoin d'Oracle : on pointe volontairement vers un port fermé."""
    monkeypatch.setenv("ORACLE_DSN", "127.0.0.1:1/inexistant")
    monkeypatch.setenv("ORACLE_CONNECT_TIMEOUT", "2")
    with pytest.raises(SearchError, match="Connexion à Oracle impossible"):
        keyword_search("SHUTDOWN", top_k=3)  # pas de connection= : ouvre la sienne