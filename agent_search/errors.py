"""Exception commune au module search (partagée par le vrai moteur et le mock).

Isolée dans son propre fichier pour éviter tout import circulaire entre
search_adapter.py (qui importe mock_search) et mock_search.py (qui a besoin
de lever la même erreur que le vrai moteur de Membre 2).
"""


class SearchError(Exception):
    """Erreur de recherche : requête invalide, backend indisponible, erreur Oracle..."""
