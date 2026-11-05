"""Tests unitaires purs (aucune connexion Oracle, aucun modèle d'embeddings).

Ces tests s'exécutent n'importe où, y compris sans accès à la base Oracle ou à
Hugging Face — ils ne testent que la logique Python pure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from search.search_utils import (
    SearchError,
    SearchResult,
    construire_requete_contains,
    formater_source,
    valider_requete,
    valider_top_k,
)


def test_valider_requete_accepte_une_requete_normale():
    assert valider_requete("  protocole DNS  ") == "protocole DNS"


def test_valider_requete_rejette_une_chaine_vide():
    with pytest.raises(SearchError):
        valider_requete("")


def test_valider_requete_rejette_des_espaces_seulement():
    with pytest.raises(SearchError):
        valider_requete("    ")


def test_valider_requete_rejette_none():
    with pytest.raises(SearchError):
        valider_requete(None)  # type: ignore[arg-type]


def test_valider_top_k_accepte_un_entier_valide():
    assert valider_top_k(5) == 5


def test_valider_top_k_rejette_zero():
    with pytest.raises(SearchError):
        valider_top_k(0)


def test_valider_top_k_rejette_negatif():
    with pytest.raises(SearchError):
        valider_top_k(-3)


def test_valider_top_k_rejette_trop_grand():
    with pytest.raises(SearchError):
        valider_top_k(1000)


def test_valider_top_k_rejette_un_flottant():
    with pytest.raises(SearchError):
        valider_top_k(2.5)  # type: ignore[arg-type]


def test_construire_requete_contains_simple():
    assert construire_requete_contains("DNS") == '"DNS"'


def test_construire_requete_contains_plusieurs_mots():
    expression = construire_requete_contains("protocole DNS")
    assert expression == '"protocole" ACCUM "DNS"'


def test_construire_requete_contains_fuzzy():
    expression = construire_requete_contains("DNS", fuzzy=True)
    assert expression == 'FUZZY("DNS")'


def test_construire_requete_contains_ignore_la_ponctuation():
    # La ponctuation ne doit pas casser l'expression Oracle Text générée.
    expression = construire_requete_contains("qu'est-ce que le DNS ?")
    assert '"DNS"' in expression
    assert "'" not in expression


def test_construire_requete_contains_chaine_vide_retourne_vide():
    assert construire_requete_contains("???") == ""


def test_formater_source_pdf_avec_page():
    assert formater_source({"source": "reseaux_osi_tcpip.pdf", "page": 4}) == (
        "reseaux_osi_tcpip.pdf (p. 4)"
    )


def test_formater_source_txt_sans_page_affichee():
    # Pour un .txt, Membre 1 stocke toujours page=1 (pas de pagination réelle) :
    # on ne l'affiche pas pour ne pas laisser croire à une vraie numérotation.
    assert formater_source({"source": "reseaux_tcp_udp.txt", "page": 1}) == (
        "reseaux_tcp_udp.txt"
    )


def test_formater_source_sans_source_connue():
    assert formater_source({}) == "document inconnu"


def test_table_name_coherente_avec_ingestion():
    """search_utils.TABLE_NAME / EMBEDDING_MODEL sont dupliquées volontairement
    (voir commentaire dans search_utils.py) pour éviter de forcer tout le module
    de recherche à dépendre de pypdf/langchain_text_splitters. Ce test vérifie
    qu'elles sont bien restées synchronisées avec database/ingestion.py, sans
    importer ce module (donc sans ses dépendances lourdes) : on lit son code
    source en texte et on y cherche les constantes NOM_TABLE / MODELE_EMBEDDING.
    """
    import re

    from search.search_utils import EMBEDDING_MODEL, TABLE_NAME

    chemin_ingestion = Path(__file__).resolve().parent.parent / "database" / "ingestion.py"
    source = chemin_ingestion.read_text(encoding="utf-8")

    m_table = re.search(r'NOM_TABLE\s*=\s*"([^"]+)"', source)
    m_modele = re.search(r'MODELE_EMBEDDING\s*=\s*"([^"]+)"', source)

    assert m_table, "NOM_TABLE introuvable dans database/ingestion.py"
    assert m_modele, "MODELE_EMBEDDING introuvable dans database/ingestion.py"
    assert TABLE_NAME == m_table.group(1), (
        "search_utils.TABLE_NAME a divergé de database.ingestion.NOM_TABLE : "
        "mettez à jour search/search_utils.py"
    )
    assert EMBEDDING_MODEL == m_modele.group(1), (
        "search_utils.EMBEDDING_MODEL a divergé de database.ingestion.MODELE_EMBEDDING : "
        "mettez à jour search/search_utils.py"
    )


def test_search_result_to_dict_contient_les_6_champs_de_linterface_membre3():
    r = SearchResult(
        chunk_id="ABC123",
        document_id="reseaux_tcp_udp.txt",
        content="Le protocole TCP...",
        source="reseaux_tcp_udp.txt",
        score=0.87,
        search_method="vector",
        metadata={"page": 1},
    )
    d = r.to_dict()
    for champ in ("chunk_id", "document_id", "content", "source", "score", "search_method"):
        assert champ in d
