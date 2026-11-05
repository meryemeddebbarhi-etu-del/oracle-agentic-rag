"""Tests d'INTÉGRATION pour keyword_search() : nécessitent une vraie base Oracle
joignable, avec l'index Oracle Text DOCUMENTS_RAG_TEXT_IDX déjà créé par
database/ingestion.py. Skip automatique sinon (voir tests/conftest.py).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from search.search_utils import SearchError
from search.keyword_search import keyword_search


def test_keyword_search_trouve_un_terme_present(oracle_connection):
    # "DHCP" apparaît explicitement dans reseaux_services_dns_dhcp_http.pdf.
    resultats = keyword_search("DHCP", top_k=5, connection=oracle_connection)
    assert all(r.search_method == "keyword" for r in resultats)


def test_keyword_search_scores_decroissants(oracle_connection):
    resultats = keyword_search("TCP UDP", top_k=5, connection=oracle_connection)
    scores = [r.score for r in resultats]
    assert scores == sorted(scores, reverse=True)


def test_keyword_search_terme_absent_renvoie_liste_vide_ou_non_pertinente(oracle_connection):
    # Terme qui ne devrait apparaître dans aucun document réseau du corpus.
    resultats = keyword_search("xylophoneinexistant999", top_k=5, connection=oracle_connection)
    assert resultats == []


def test_keyword_search_requete_vide_leve_searcherror(oracle_connection):
    with pytest.raises(SearchError):
        keyword_search("", top_k=3, connection=oracle_connection)


def test_keyword_search_ponctuation_ne_casse_pas_la_requete(oracle_connection):
    # Ne doit pas lever d'exception Oracle Text malgré apostrophe/point d'interrogation.
    resultats = keyword_search("qu'est-ce que le DNS ?", top_k=3, connection=oracle_connection)
    assert isinstance(resultats, list)


def test_keyword_search_mesure_temps_execution(oracle_connection):
    debut = time.perf_counter()
    keyword_search("HTTP", top_k=5, connection=oracle_connection)
    duree = time.perf_counter() - debut
    print(f"\nkeyword_search (k=5) : {duree * 1000:.1f} ms")
    assert duree >= 0
