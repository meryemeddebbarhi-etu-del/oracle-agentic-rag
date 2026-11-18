"""Mesure la qualité de la recherche sur les questions de test.

Pour chaque question, on cherche le RANG du premier morceau qui contient le
marqueur attendu (une phrase ou un mot qui n'existe que dans le bon passage).
Le script ne modifie RIEN dans la base : il ne fait que lire.

Lancement (VM allumée), depuis la racine du projet :
    python evaluation_recherche.py
"""

from search import hybrid_search, keyword_search, vector_search
from search.search_utils import get_connection

TOP_K = 13  # = nombre total de morceaux dans la base de test

# (question, marqueur présent uniquement dans le bon morceau, fichier attendu)
QUESTIONS = [
    ("Qu'est-ce que le three-way handshake ?", "three-way", "reseaux_tcp_udp.txt"),
    ("Combien de couches a le modèle OSI ?", "sept couches", "reseaux_osi_tcpip.pdf"),
    ("Sur quelle couche travaille un routeur ?", "à la couche 3", "reseaux_osi_tcpip.pdf"),
    ("Combien de bits font une adresse IPv4 et une adresse IPv6 ?", "128 bits", "reseaux_osi_tcpip.pdf"),
    ("Que signifie DORA dans le DHCP ?", "DORA", "reseaux_services_dns_dhcp_http.pdf"),
    ("Quels ports utilise le DHCP ?", "port 67", "reseaux_services_dns_dhcp_http.pdf"),
    ("Quel code HTTP indique qu'une ressource n'existe pas ?", "404", "reseaux_services_dns_dhcp_http.pdf"),
    ("Quel port utilise SSH ?", "SSH (connexion", "reseaux_services_dns_dhcp_http.pdf"),
]

QUESTIONS_SANS_REPONSE = [
    "Quel est le prix d'un routeur Cisco ?",
    "Comment configurer un VLAN sur un switch ?",
]


def rang(resultats, marqueur):
    """Rang (à partir de 1) du premier résultat contenant le marqueur, sinon None."""
    for i, r in enumerate(resultats, start=1):
        # Les sauts de ligne issus des PDF peuvent couper un marqueur en deux
        # ("sept\ncouches") : on les ramène à de simples espaces avant de comparer.
        if marqueur in " ".join(r.content.split()):
            return i
    return None


def texte_rang(valeur):
    return "absent" if valeur is None else str(valeur)


if __name__ == "__main__":
    conn = get_connection()
    lignes = []
    for question, marqueur, _fichier in QUESTIONS:
        v = rang(vector_search(question, top_k=TOP_K, connection=conn), marqueur)
        m = rang(keyword_search(question, top_k=TOP_K, connection=conn), marqueur)
        h = rang(hybrid_search(question, top_k=TOP_K, connection=conn), marqueur)
        lignes.append((question, v, m, h))

    print()
    print(f"{'Question':58} {'vecteur':>8} {'mots-clés':>10} {'hybride':>8}")
    print("-" * 88)
    for question, v, m, h in lignes:
        print(f"{question[:57]:58} {texte_rang(v):>8} {texte_rang(m):>10} {texte_rang(h):>8}")

    print()
    for nom, indice in (("vecteur", 1), ("mots-clés", 2), ("hybride", 3)):
        rangs = [ligne[indice] for ligne in lignes]
        premier = sum(1 for x in rangs if x == 1)
        top3 = sum(1 for x in rangs if x is not None and x <= 3)
        top5 = sum(1 for x in rangs if x is not None and x <= 5)
        print(f"{nom:10} : 1er rang {premier}/{len(rangs)}, dans le top 3 {top3}/{len(rangs)}, dans le top 5 {top5}/{len(rangs)}")

    print()
    print("Questions SANS réponse dans les documents (hybride, 3 premiers résultats) :")
    for question in QUESTIONS_SANS_REPONSE:
        print(" -", question)
        for r in hybrid_search(question, top_k=3, connection=conn):
            print(f"     {r.score:.4f}  {r.source}")

    conn.close()