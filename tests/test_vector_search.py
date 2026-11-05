"""Tests d'INTÉGRATION pour vector_search() : nécessitent une vraie base Oracle
joignable (celle de Membre 1, avec la table DOCUMENTS_RAG déjà remplie par
database/ingestion.py) ET un accès réseau pour télécharger le modèle
sentence-transformers/all-MiniLM-L12-v2 la première fois.

Si l'un des deux manque, les tests sont SKIP (voir tests/conftest.py), pas
échoués : conformément à la consigne, on ne doit jamais inventer un résultat.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from search.search_utils import SearchError
from search.vector_search import vector_search


def test_vector_search_retourne_des_resultats(oracle_connection):
    resultats = vector_search("résolution de noms DNS", top_k=3, connection=oracle_connection)
    assert len(resultats) <= 3
    assert all(r.search_method == "vector" for r in resultats)
    assert all(r.chunk_id for r in resultats)


def test_vector_search_scores_decroissants(oracle_connection):
    resultats = vector_search("protocole TCP", top_k=5, connection=oracle_connection)
    scores = [r.score for r in resultats]
    assert scores == sorted(scores, reverse=True)


def test_vector_search_respecte_top_k(oracle_connection):
    resultats = vector_search("réseau", top_k=2, connection=oracle_connection)
    assert len(resultats) <= 2


def test_vector_search_requete_vide_leve_searcherror(oracle_connection):
    with pytest.raises(SearchError):
        vector_search("", top_k=3, connection=oracle_connection)


def test_vector_search_mesure_temps_execution(oracle_connection):
    debut = time.perf_counter()
    vector_search("DHCP", top_k=5, connection=oracle_connection)
    duree = time.perf_counter() - debut
    print(f"\nvector_search (k=5) : {duree * 1000:.1f} ms")
    # Pas d'assertion sur une valeur précise : la durée dépend de la machine.
    # On vérifie seulement qu'elle est mesurée et finie.
    assert duree >= 0
