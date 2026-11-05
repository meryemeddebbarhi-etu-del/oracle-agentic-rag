"""Module de recherche (Membre 2) : vectorielle, mots-clés, hybride.

Interface publique pour Membre 3 (agent LangGraph) :

    from search import vector_search, keyword_search, hybrid_search, SearchError

    resultats = hybrid_search("comment fonctionne DHCP ?", top_k=5)
    for r in resultats:
        print(r.source, r.score, r.content[:100])

Chaque fonction renvoie une liste de `SearchResult` (voir search_utils.py) :
mêmes champs quelle que soit la méthode (chunk_id, document_id, content,
source, score, search_method, metadata), pour que Membre 3 n'ait jamais à
distinguer les trois modes dans son code.
"""

from .hybrid_search import hybrid_search
from .keyword_search import keyword_search
from .search_utils import SearchError, SearchResult
from .vector_search import vector_search

__all__ = [
    "vector_search",
    "keyword_search",
    "hybrid_search",
    "SearchResult",
    "SearchError",
]
