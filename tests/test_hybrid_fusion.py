"""Teste la logique de fusion RRF de hybrid_search(), SANS toucher Oracle ni
Hugging Face : vector_search et keyword_search sont remplacés par des doubles
de test (monkeypatch) qui renvoient des listes fixes de SearchResult.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib

# search/__init__.py fait `from .hybrid_search import hybrid_search`, ce qui
# réécrit l'attribut `search.hybrid_search` (il pointe vers la FONCTION, pas le
# sous-module) : `import search.hybrid_search as x` récupérerait donc la
# fonction, pas le module, et monkeypatch n'aurait rien à patcher dessus.
# importlib.import_module() renvoie toujours le vrai objet module.
hybrid_search_module = importlib.import_module("search.hybrid_search")
from search.search_utils import SearchResult


class _FakeConn:
    """Connexion factice : seule méthode utilisée par hybrid_search() ici est close()."""

    def close(self):
        pass


def _sr(chunk_id, method, score, source="doc.txt"):
    return SearchResult(
        chunk_id=chunk_id,
        document_id=source,
        content=f"contenu de {chunk_id}",
        source=source,
        score=score,
        search_method=method,
        metadata={},
    )


def test_hybrid_search_fusionne_sans_doublon(monkeypatch):
    # "A" est 1er en vectoriel ET 2e en mots-clés -> doit apparaître UNE SEULE fois,
    # avec un score RRF qui cumule les deux contributions.
    vecteur = [_sr("A", "vector", 0.9), _sr("B", "vector", 0.8)]
    motscles = [_sr("C", "keyword", 55.0), _sr("A", "keyword", 40.0)]

    monkeypatch.setattr(hybrid_search_module, "vector_search", lambda q, top_k, connection=None: vecteur)
    monkeypatch.setattr(hybrid_search_module, "keyword_search", lambda q, top_k, connection=None: motscles)
    monkeypatch.setattr(hybrid_search_module, "get_connection", lambda: _FakeConn())

    resultats = hybrid_search_module.hybrid_search("peu importe", top_k=10, candidate_k=10)

    ids = [r.chunk_id for r in resultats]
    assert ids.count("A") == 1, "le chunk présent dans les deux listes ne doit apparaître qu'une fois"
    assert set(ids) == {"A", "B", "C"}
    assert all(r.search_method == "hybrid" for r in resultats)


def test_hybrid_search_score_rrf_est_correct(monkeypatch):
    # A : rang 1 vectoriel, rang 2 mots-clés -> rrf = 1/(60+1) + 1/(60+2)
    vecteur = [_sr("A", "vector", 0.9), _sr("B", "vector", 0.8)]
    motscles = [_sr("C", "keyword", 55.0), _sr("A", "keyword", 40.0)]

    monkeypatch.setattr(hybrid_search_module, "vector_search", lambda q, top_k, connection=None: vecteur)
    monkeypatch.setattr(hybrid_search_module, "keyword_search", lambda q, top_k, connection=None: motscles)
    monkeypatch.setattr(hybrid_search_module, "get_connection", lambda: _FakeConn())

    resultats = hybrid_search_module.hybrid_search("peu importe", top_k=10, candidate_k=10, rrf_k=60)
    par_id = {r.chunk_id: r for r in resultats}

    attendu_a = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert abs(par_id["A"].score - attendu_a) < 1e-9

    attendu_b = 1.0 / (60 + 2)  # B seulement vectoriel, rang 2
    assert abs(par_id["B"].score - attendu_b) < 1e-9


def test_hybrid_search_tri_par_score_decroissant(monkeypatch):
    vecteur = [_sr("A", "vector", 0.9), _sr("B", "vector", 0.8), _sr("D", "vector", 0.1)]
    motscles = [_sr("C", "keyword", 55.0), _sr("A", "keyword", 40.0)]

    monkeypatch.setattr(hybrid_search_module, "vector_search", lambda q, top_k, connection=None: vecteur)
    monkeypatch.setattr(hybrid_search_module, "keyword_search", lambda q, top_k, connection=None: motscles)
    monkeypatch.setattr(hybrid_search_module, "get_connection", lambda: _FakeConn())

    resultats = hybrid_search_module.hybrid_search("peu importe", top_k=10, candidate_k=10)
    scores = [r.score for r in resultats]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_respecte_top_k(monkeypatch):
    vecteur = [_sr(str(i), "vector", 1.0 - i * 0.1) for i in range(10)]
    motscles = []

    monkeypatch.setattr(hybrid_search_module, "vector_search", lambda q, top_k, connection=None: vecteur)
    monkeypatch.setattr(hybrid_search_module, "keyword_search", lambda q, top_k, connection=None: motscles)
    monkeypatch.setattr(hybrid_search_module, "get_connection", lambda: _FakeConn())

    resultats = hybrid_search_module.hybrid_search("peu importe", top_k=3, candidate_k=10)
    assert len(resultats) == 3


def test_hybrid_search_poids_favorise_la_methode_voulue(monkeypatch):
    # Un seul document, uniquement trouvé par mots-clés : avec w_keyword=0 il ne
    # doit avoir aucune contribution de ce côté (mais reste présent, car RRF
    # fusionne l'UNION des deux listes, pas leur intersection).
    vecteur = []
    motscles = [_sr("C", "keyword", 55.0)]

    monkeypatch.setattr(hybrid_search_module, "vector_search", lambda q, top_k, connection=None: vecteur)
    monkeypatch.setattr(hybrid_search_module, "keyword_search", lambda q, top_k, connection=None: motscles)
    monkeypatch.setattr(hybrid_search_module, "get_connection", lambda: _FakeConn())

    resultats = hybrid_search_module.hybrid_search(
        "peu importe", top_k=10, candidate_k=10, w_vector=2.0, w_keyword=0.0
    )
    assert resultats[0].score == 0.0
