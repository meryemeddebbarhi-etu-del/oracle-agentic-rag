"""Moteur de recherche simulé (MockSearch) — permet de développer et tester
l'agent du Membre 3 sans attendre le vrai moteur du Membre 2, et sans
connexion Oracle.

Le corpus simulé reprend volontairement les MÊMES thèmes que les documents
réels du projet (réseaux : OSI, TCP/IP, DNS, DHCP, HTTP, UDP), pour que les
démonstrations et captures d'écran de soutenance restent cohérentes avec le
travail des autres membres, même quand ce module tourne seul.

Chaque fonction a EXACTEMENT la même signature et renvoie EXACTEMENT le même
format de résultat (liste de dict avec les clés chunk_id, document_id,
content, source, score, search_method, metadata) que le vrai module
search/ du Membre 2 (voir search/search_utils.py::SearchResult.to_dict()).
Remplacer MockSearchAdapter par RealSearchAdapter dans search_adapter.py ne
change donc rien au reste du code de l'agent.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

from .errors import SearchError

# ---------------------------------------------------------------------------
# Corpus simulé : mêmes thématiques que documents/ (Membre 1)
# ---------------------------------------------------------------------------

_CORPUS: list[dict[str, Any]] = [
    {
        "chunk_id": "mock-001",
        "document_id": "reseaux_osi_tcpip.pdf",
        "source": "reseaux_osi_tcpip.pdf (p. 1)",
        "metadata": {"page": 1, "categorie": "modele_osi"},
        "content": (
            "Le modele OSI (Open Systems Interconnection) decoupe les "
            "communications reseau en sept couches : physique, liaison de "
            "donnees, reseau, transport, session, presentation et "
            "application. Chaque couche ne communique qu'avec les couches "
            "immediatement adjacentes, ce qui permet de faire evoluer une "
            "couche sans casser les autres."
        ),
    },
    {
        "chunk_id": "mock-002",
        "document_id": "reseaux_osi_tcpip.pdf",
        "source": "reseaux_osi_tcpip.pdf (p. 2)",
        "metadata": {"page": 2, "categorie": "modele_tcpip"},
        "content": (
            "Le modele TCP/IP, plus utilise en pratique qu'OSI, regroupe les "
            "fonctions en quatre couches : acces reseau, internet, "
            "transport et application. La couche internet est assuree par "
            "le protocole IP, responsable de l'adressage et du routage des "
            "paquets entre reseaux."
        ),
    },
    {
        "chunk_id": "mock-003",
        "document_id": "reseaux_tcp_udp.txt",
        "source": "reseaux_tcp_udp.txt",
        "metadata": {"categorie": "transport"},
        "content": (
            "TCP (Transmission Control Protocol) est un protocole de la "
            "couche transport oriente connexion : il garantit la livraison "
            "ordonnee des segments grace a un mecanisme d'accuse de "
            "reception et de retransmission. UDP (User Datagram Protocol), "
            "au contraire, n'offre aucune garantie de livraison ni d'ordre, "
            "mais il est plus rapide et plus leger, ce qui le rend adapte "
            "au streaming video ou a la voix sur IP."
        ),
    },
    {
        "chunk_id": "mock-004",
        "document_id": "reseaux_services_dns_dhcp_http.pdf",
        "source": "reseaux_services_dns_dhcp_http.pdf (p. 1)",
        "metadata": {"page": 1, "categorie": "dns"},
        "content": (
            "Le protocole DNS (Domain Name System) traduit un nom de "
            "domaine lisible par un humain, comme exemple.com, en une "
            "adresse IP exploitable par les routeurs. Une requete DNS "
            "parcourt generalement un serveur recursif, un serveur racine, "
            "un serveur de domaine de premier niveau, puis le serveur "
            "faisant autorite pour le domaine demande."
        ),
    },
    {
        "chunk_id": "mock-005",
        "document_id": "reseaux_services_dns_dhcp_http.pdf",
        "source": "reseaux_services_dns_dhcp_http.pdf (p. 2)",
        "metadata": {"page": 2, "categorie": "dhcp"},
        "content": (
            "Le protocole DHCP (Dynamic Host Configuration Protocol) "
            "attribue automatiquement une adresse IP, un masque de "
            "sous-reseau, une passerelle et des serveurs DNS a chaque "
            "machine qui rejoint le reseau. L'echange se deroule en quatre "
            "etapes connues sous le nom de DORA : Discover, Offer, "
            "Request, Acknowledge."
        ),
    },
    {
        "chunk_id": "mock-006",
        "document_id": "reseaux_services_dns_dhcp_http.pdf",
        "source": "reseaux_services_dns_dhcp_http.pdf (p. 3)",
        "metadata": {"page": 3, "categorie": "http"},
        "content": (
            "HTTP (HyperText Transfer Protocol) est le protocole de la "
            "couche application utilise par le web. Il fonctionne selon un "
            "modele requete-reponse sans etat : chaque requete du client "
            "est independante des precedentes. HTTPS ajoute une couche de "
            "chiffrement TLS pour proteger les echanges entre le "
            "navigateur et le serveur."
        ),
    },
]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Mots vides français : ignorés dans les deux recherches simulées pour éviter
# qu'une question totalement hors sujet (ex. "Quelle est la recette de la
# tarte Tatin ?") ne décroche un score artificiellement élevé simplement
# parce qu'elle partage des mots de liaison très fréquents avec le corpus.
_MOTS_VIDES = {
    "le", "la", "les", "l", "un", "une", "des", "de", "du", "d", "au", "aux",
    "et", "ou", "est", "sont", "ete", "etre", "a", "ai", "as", "avons", "avez", "ont",
    "ce", "ces", "cet", "cette", "qui", "que", "quoi", "quel", "quelle", "quels", "quelles",
    "comment", "pourquoi", "quand", "ou", "dans", "sur", "sous", "par", "pour", "avec",
    "sans", "entre", "vers", "chez", "mais", "donc", "or", "ni", "car",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "mon", "ma", "mes",
    "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
    "se", "si", "ne", "pas", "plus", "tres", "fait", "faire", "peut", "peuvent",
}


def _normaliser(texte: str) -> str:
    """Minuscules + suppression des accents, pour des comparaisons robustes."""
    texte = texte.lower()
    texte = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in texte if not unicodedata.combining(c))


def _valider_requete(query: str) -> str:
    if query is None:
        raise SearchError("La requête ne peut pas être vide (None).")
    query = query.strip()
    if not query:
        raise SearchError("La requête ne peut pas être vide.")
    return query


def _valider_top_k(top_k: int) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise SearchError("top_k doit être un entier.")
    if top_k < 1:
        raise SearchError("top_k doit être supérieur ou égal à 1.")
    if top_k > 200:
        raise SearchError("top_k est trop grand (maximum 200).")
    return top_k


def _resultat(doc: dict[str, Any], score: float, method: str, extra_meta: dict) -> dict[str, Any]:
    metadata = dict(doc["metadata"])
    metadata.update(extra_meta)
    metadata["mock_mode"] = True
    return {
        "chunk_id": doc["chunk_id"],
        "document_id": doc["document_id"],
        "content": doc["content"],
        "source": doc["source"],
        "score": round(float(score), 4),
        "search_method": method,
        "metadata": metadata,
    }


def vector_search(query: str, top_k: int = 5, **_kwargs) -> list[dict[str, Any]]:
    """Simule une recherche sémantique par chevauchement de mots normalisés.

    Ce n'est PAS une vraie similarité d'embeddings : c'est une approximation
    volontairement simple (score dans [0, 1]) qui suffit pour développer et
    tester la logique de l'agent (choix de méthode, formatage du contexte,
    détection d'absence de réponse) sans dépendre d'un modèle d'embeddings
    ni d'Oracle.
    """
    query = _valider_requete(query)
    top_k = _valider_top_k(top_k)

    mots_requete = set(_TOKEN_RE.findall(_normaliser(query))) - _MOTS_VIDES
    resultats = []
    for doc in _CORPUS:
        mots_doc = set(_TOKEN_RE.findall(_normaliser(doc["content"]))) - _MOTS_VIDES
        if not mots_requete or not mots_doc:
            score = 0.0
        else:
            intersection = mots_requete & mots_doc
            score = len(intersection) / len(mots_requete)
        resultats.append(_resultat(doc, score, "vector", {"simulated_vector_score": score}))

    resultats.sort(key=lambda r: r["score"], reverse=True)
    return resultats[:top_k]


def keyword_search(query: str, top_k: int = 5, *, fuzzy: bool = False, **_kwargs) -> list[dict[str, Any]]:
    """Simule une recherche par mots-clés exacts (équivalent du CONTAINS Oracle Text).

    Score = nombre d'occurrences totales des termes de la requête dans le
    document, à l'échelle d'un entier (comme le SCORE() d'Oracle Text),
    jamais dans la même échelle que vector_search.
    """
    query = _valider_requete(query)
    top_k = _valider_top_k(top_k)

    mots_requete = [m for m in _TOKEN_RE.findall(_normaliser(query)) if m not in _MOTS_VIDES]
    resultats = []
    for doc in _CORPUS:
        texte_norm = _normaliser(doc["content"])
        occurrences = 0
        for mot in mots_requete:
            if fuzzy:
                # Tolérance simple aux fautes de frappe : préfixe de 4 lettres.
                prefixe = mot[:4] if len(mot) >= 4 else mot
                occurrences += texte_norm.count(prefixe)
            else:
                occurrences += len(re.findall(rf"\b{re.escape(mot)}\b", texte_norm))
        score = occurrences * 10  # mise à l'échelle grossière façon Oracle Text
        if score > 0:
            resultats.append(_resultat(doc, score, "keyword", {"occurrences": occurrences}))

    resultats.sort(key=lambda r: r["score"], reverse=True)
    return resultats[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 5,
    *,
    candidate_k: Optional[int] = None,
    rrf_k: int = 60,
    w_vector: float = 1.0,
    w_keyword: float = 1.0,
    **_kwargs,
) -> list[dict[str, Any]]:
    """Fusionne vector_search et keyword_search par Reciprocal Rank Fusion (RRF),
    exactement selon le même principe que search/hybrid_search.py du Membre 2 :
    seul le RANG de chaque document dans chaque liste compte, pas son score brut
    (les deux échelles de score ne sont pas comparables).
    """
    query = _valider_requete(query)
    top_k = _valider_top_k(top_k)
    candidate_k = candidate_k or max(20, top_k * 4)

    # On exclut les candidats à score strictement nul : un score de 0 signifie
    # "aucun mot significatif en commun" (vector) ou "aucune occurrence"
    # (keyword), donc ce candidat n'a aucune pertinence réelle et ne doit pas
    # recevoir un score RRF non nul simplement parce qu'il occupe un rang.
    vec_resultats = [r for r in vector_search(query, top_k=candidate_k) if r["score"] > 0]
    kw_resultats = [r for r in keyword_search(query, top_k=candidate_k) if r["score"] > 0]

    rang_vec = {r["chunk_id"]: i + 1 for i, r in enumerate(vec_resultats)}
    rang_kw = {r["chunk_id"]: i + 1 for i, r in enumerate(kw_resultats)}
    par_id = {r["chunk_id"]: r for r in vec_resultats}
    par_id.update({r["chunk_id"]: r for r in kw_resultats})

    scores_rrf: dict[str, float] = {}
    for chunk_id in set(rang_vec) | set(rang_kw):
        score = 0.0
        if chunk_id in rang_vec:
            score += w_vector * (1.0 / (rrf_k + rang_vec[chunk_id]))
        if chunk_id in rang_kw:
            score += w_keyword * (1.0 / (rrf_k + rang_kw[chunk_id]))
        scores_rrf[chunk_id] = score

    fusionnes = []
    for chunk_id, score_rrf in sorted(scores_rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        base = par_id[chunk_id]
        metadata = dict(base["metadata"])
        metadata["rrf_score"] = round(score_rrf, 6)
        metadata["rank_vector"] = rang_vec.get(chunk_id)
        metadata["rank_keyword"] = rang_kw.get(chunk_id)
        fusionnes.append(
            {
                "chunk_id": chunk_id,
                "document_id": base["document_id"],
                "content": base["content"],
                "source": base["source"],
                "score": round(score_rrf, 6),
                "search_method": "hybrid",
                "metadata": metadata,
            }
        )
    return fusionnes
