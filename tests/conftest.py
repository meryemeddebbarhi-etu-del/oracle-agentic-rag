"""Fixtures partagées pour les tests.

`oracle_connection` ouvre une vraie connexion Oracle pour les tests
d'intégration. Si la base n'est pas joignable (pas de VM/Docker Oracle démarré,
ou exécution dans un environnement sans accès réseau à la base), le test est
SKIP (pas échoué) avec un message explicite — conformément à la consigne de ne
jamais inventer un résultat de test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Permet d'exécuter `pytest` depuis la racine du projet sans installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def oracle_connection():
    from search.search_utils import SearchError, get_connection

    try:
        conn = get_connection()
    except SearchError as erreur:
        pytest.skip(f"Oracle Database non joignable, test d'intégration ignoré : {erreur}")
    yield conn
    conn.close()
