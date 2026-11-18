"""Tests d'INTÉGRATION de hybrid_search() de bout en bout (Membre 2) : vraie base
Oracle + vrai modèle d'embeddings. Pour la LOGIQUE de fusion RRF seule (sans
Oracle), voir tests/test_hybrid_fusion.py : ceux-là tournent partout.

Sans base joignable, la fixture `oracle_connection` SKIP le test.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from search.hybrid_search import hybrid_search
from search.search_utils import SearchError, get_embeddings
from search.keyword_search import keyword_search
from search.vector_search import vector_search

DUREE_MAX_S = float(os.getenv("ORACLE_TEST_MAX_SECONDS", "10"))
RRF_MAX = 2.0 / 61  # un morceau classé 1er par les deux méthodes (rrf_k = 60)


@pytest.fixture(scope="module")
def modele_charge():
    return get_embeddings()


def _documents(resultats):
    return {r.document_id for r in resultats}


@pytest.mark.parametrize(
    "question, document_attendu",
    [
        ("Quels sont les modes SHUTDOWN ?", "oracle_demarrage_arret_parametres.txt"),
        ("Comment utiliser EXECUTE IMMEDIATE ?", "plsql_sql_dynamique_metadata.txt"),
        ("Comment exporter avec expdp ?", "oracle_datapump_sqlloader.txt"),
        ("Qu'est-ce qu'un rôle ?", "oracle_utilisateurs_privileges_roles_profils.txt"),
    ],
)
def test_hybrid_search_retrouve_le_bon_document(oracle_connection, modele_charge, question, document_attendu):
    resultats = hybrid_search(question, top_k=5, connection=oracle_connection)
    assert document_attendu in _documents(resultats)


def test_hybrid_search_format_et_scores_rrf(oracle_connection, modele_charge):
    resultats = hybrid_search("Quels sont les modes SHUTDOWN ?", top_k=5, connection=oracle_connection)
    assert 1 <= len(resultats) <= 5
    scores = [r.score for r in resultats]
    assert scores == sorted(scores, reverse=True)
    for r in resultats:
        assert r.search_method == "hybrid"
        assert 0 < r.score <= RRF_MAX + 1e-9
        assert r.metadata["rrf_k"] == 60
        # chaque résultat vient d'au moins une des deux méthodes
        assert "vector_rank" in r.metadata or "keyword_rank" in r.metadata


def test_hybrid_search_pas_de_doublons_et_respecte_top_k(oracle_connection, modele_charge):
    resultats = hybrid_search("PDB tablespace utilisateur listener", top_k=10, connection=oracle_connection)
    ids = [r.chunk_id for r in resultats]
    assert len(ids) == len(set(ids)), "aucun morceau ne doit apparaître deux fois"
    assert len(ids) <= 10


def test_hybrid_search_le_premier_resultat_vient_des_deux_sous_recherches(oracle_connection, modele_charge):
    question = "Quels sont les modes SHUTDOWN ?"
    vecteur = {r.chunk_id for r in vector_search(question, top_k=20, connection=oracle_connection)}
    motscles = {r.chunk_id for r in keyword_search(question, top_k=20, connection=oracle_connection)}
    premier = hybrid_search(question, top_k=1, connection=oracle_connection)[0]
    assert premier.chunk_id in (vecteur | motscles)


def test_hybrid_search_un_morceau_trouve_par_les_deux_methodes_passe_devant(oracle_connection, modele_charge):
    resultats = hybrid_search("SHUTDOWN IMMEDIATE", top_k=10, connection=oracle_connection)
    deux = [r for r in resultats if "vector_rank" in r.metadata and "keyword_rank" in r.metadata]
    une = [r for r in resultats if ("vector_rank" in r.metadata) != ("keyword_rank" in r.metadata)]
    if deux and une:
        assert max(r.score for r in deux) >= max(r.score for r in une)


@pytest.mark.parametrize("requete", ["", "   ", None])
def test_hybrid_search_requete_vide_leve_searcherror(oracle_connection, requete):
    with pytest.raises(SearchError):
        hybrid_search(requete, top_k=3, connection=oracle_connection)


@pytest.mark.parametrize("top_k", [0, 201, "3"])
def test_hybrid_search_top_k_invalide_leve_searcherror(oracle_connection, top_k):
    with pytest.raises(SearchError):
        hybrid_search("SHUTDOWN", top_k=top_k, connection=oracle_connection)


def test_hybrid_search_est_assez_rapide(oracle_connection, modele_charge):
    hybrid_search("warm-up", top_k=1, connection=oracle_connection)
    debut = time.perf_counter()
    hybrid_search("Qu'est-ce que le multitenant ?", top_k=5, connection=oracle_connection)
    duree = time.perf_counter() - debut
    print(f"\nhybrid_search (k=5) : {duree * 1000:.1f} ms")
    assert duree < DUREE_MAX_S, f"trop lent : {duree:.1f} s (seuil {DUREE_MAX_S} s)"


def test_hybrid_search_base_injoignable_leve_searcherror(monkeypatch):
    """N'a PAS besoin d'Oracle : on pointe volontairement vers un port fermé."""
    monkeypatch.setenv("ORACLE_DSN", "127.0.0.1:1/inexistant")
    monkeypatch.setenv("ORACLE_CONNECT_TIMEOUT", "2")
    with pytest.raises(SearchError, match="Connexion à Oracle impossible"):
        hybrid_search("SHUTDOWN", top_k=3)