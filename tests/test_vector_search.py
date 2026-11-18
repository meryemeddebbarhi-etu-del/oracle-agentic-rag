"""Tests d'INTÉGRATION de vector_search() (Membre 2).

Ils nécessitent une vraie base Oracle avec la table DOCUMENTS_RAG remplie par
database/ingestion.py (corpus : documents/*.txt, sujets Oracle et PL/SQL) et le
modèle d'embeddings. Si la base n'est pas joignable, la fixture `oracle_connection`
(tests/conftest.py) SKIP le test au lieu de le faire échouer.

Contrairement à l'ancienne version, chaque test vérifie un comportement réel :
on pose une question dont on connaît le document de réponse dans documents/ et
on vérifie que ce document remonte.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from search.search_utils import SearchError, get_embeddings
from search.vector_search import vector_search

DUREE_MAX_S = float(os.getenv("ORACLE_TEST_MAX_SECONDS", "10"))


@pytest.fixture(scope="module")
def modele_charge():
    """Charge le modèle d'embeddings une seule fois (le 1er chargement est lent)."""
    return get_embeddings()


def _documents(resultats):
    return {r.document_id for r in resultats}


@pytest.mark.parametrize(
    "question, document_attendu",
    [
        ("Quels sont les modes d'arrêt d'une instance ?", "oracle_demarrage_arret_parametres.txt"),
        ("Comment gérer les exceptions en PL/SQL ?", "plsql_exceptions.txt"),
        ("Comment créer une fonction stockée ?", "plsql_fonctions_stockees.txt"),
        ("À quoi sert un tablespace ?", "oracle_stockage_tablespaces.txt"),
        ("Comment configurer le listener Oracle Net ?", "oracle_net_listener.txt"),
        ("Comment importer des données avec Data Pump ?", "oracle_datapump_sqlloader.txt"),
    ],
)
def test_vector_search_retrouve_le_bon_document(oracle_connection, modele_charge, question, document_attendu):
    resultats = vector_search(question, top_k=5, connection=oracle_connection)
    assert document_attendu in _documents(resultats), (
        f"'{document_attendu}' absent du top 5 pour : {question} (obtenu : {sorted(_documents(resultats))})"
    )


def test_vector_search_format_des_resultats(oracle_connection, modele_charge):
    resultats = vector_search("Qu'est-ce qu'un rôle ?", top_k=3, connection=oracle_connection)
    assert 1 <= len(resultats) <= 3
    for r in resultats:
        assert r.search_method == "vector"
        assert r.chunk_id and r.content.strip()
        assert r.document_id.endswith(".txt")
        assert r.source
        assert -1.0 <= r.score <= 1.0 + 1e-6
        assert "distance" in r.metadata
        assert set(r.to_dict()) == {
            "chunk_id", "document_id", "content", "source", "score", "search_method", "metadata",
        }


def test_vector_search_scores_decroissants_et_sans_doublon(oracle_connection, modele_charge):
    resultats = vector_search("sauvegarde et paramètres du SPFILE", top_k=8, connection=oracle_connection)
    scores = [r.score for r in resultats]
    assert scores == sorted(scores, reverse=True)
    ids = [r.chunk_id for r in resultats]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("k", [1, 2, 7])
def test_vector_search_respecte_top_k(oracle_connection, modele_charge, k):
    resultats = vector_search("base de données Oracle", top_k=k, connection=oracle_connection)
    assert len(resultats) == k  # le corpus contient largement plus de 7 morceaux


def test_vector_search_question_hors_sujet_a_un_score_plus_faible(oracle_connection, modele_charge):
    # C'est ce qui permet à l'agent de détecter "aucune réponse dans les documents".
    sur_le_sujet = vector_search("Quels sont les modes SHUTDOWN ?", top_k=1, connection=oracle_connection)[0]
    hors_sujet = vector_search("recette traditionnelle du couscous marocain", top_k=1, connection=oracle_connection)[0]
    assert hors_sujet.score < sur_le_sujet.score


@pytest.mark.parametrize("requete", ["", "   ", None])
def test_vector_search_requete_vide_leve_searcherror(oracle_connection, requete):
    with pytest.raises(SearchError):
        vector_search(requete, top_k=3, connection=oracle_connection)


@pytest.mark.parametrize("top_k", [0, -1, 201, "5", True, 2.5])
def test_vector_search_top_k_invalide_leve_searcherror(oracle_connection, top_k):
    with pytest.raises(SearchError):
        vector_search("DHCP", top_k=top_k, connection=oracle_connection)


def test_vector_search_est_assez_rapide(oracle_connection, modele_charge):
    vector_search("warm-up", top_k=1, connection=oracle_connection)  # amorce l'index vectoriel
    debut = time.perf_counter()
    vector_search("Qu'est-ce qu'une PDB ?", top_k=5, connection=oracle_connection)
    duree = time.perf_counter() - debut
    print(f"\nvector_search (k=5) : {duree * 1000:.1f} ms")
    assert duree < DUREE_MAX_S, f"trop lent : {duree:.1f} s (seuil {DUREE_MAX_S} s)"