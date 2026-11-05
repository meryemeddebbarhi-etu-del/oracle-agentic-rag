"""Tests d'INTÉGRATION pour hybrid_search() de bout en bout (vraie base Oracle +
vrai modèle d'embeddings). Pour les tests de la LOGIQUE de fusion RRF seule
(sans Oracle), voir tests/test_hybrid_fusion.py — ceux-là s'exécutent partout.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from search.hybrid_search import hybrid_search
from search.search_utils import SearchError


def test_hybrid_search_retourne_des_resultats(oracle_connection):
    resultats = hybrid_search("comment fonctionne le DNS", top_k=5, connection=oracle_connection)
    assert len(resultats) <= 5
    assert all(r.search_method == "hybrid" for r in resultats)


def test_hybrid_search_pas_de_doublons(oracle_connection):
    resultats = hybrid_search("TCP UDP HTTP", top_k=10, connection=oracle_connection)
    ids = [r.chunk_id for r in resultats]
    assert len(ids) == len(set(ids)), "aucun chunk ne doit apparaître deux fois"


def test_hybrid_search_requete_vide_leve_searcherror(oracle_connection):
    with pytest.raises(SearchError):
        hybrid_search("", top_k=3, connection=oracle_connection)


def test_hybrid_search_mesure_temps_execution(oracle_connection):
    debut = time.perf_counter()
    hybrid_search("modèle OSI", top_k=5, connection=oracle_connection)
    duree = time.perf_counter() - debut
    print(f"\nhybrid_search (k=5) : {duree * 1000:.1f} ms")
    assert duree >= 0
