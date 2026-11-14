"""Interface de recherche utilisée par l'agent (Membre 3).

    from search import get_search_adapter, SearchError

    adapter = get_search_adapter()          # mock ou real selon .env
    resultats = adapter.hybrid_search("comment fonctionne DHCP ?", top_k=5)
"""

from .errors import SearchError
from .search_adapter import MockSearchAdapter, RealSearchAdapter, SearchAdapter, get_search_adapter

__all__ = [
    "SearchAdapter",
    "MockSearchAdapter",
    "RealSearchAdapter",
    "get_search_adapter",
    "SearchError",
]
