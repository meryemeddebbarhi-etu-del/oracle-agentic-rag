"""Teste la lecture : pose les questions de test et affiche les 3 meilleurs morceaux.

Ne modifie rien dans la base.
"""

from connexion_lecture import ouvrir_vector_store

QUESTIONS = [
    "Qu'est-ce que le three-way handshake ?",
    "Combien de couches a le modèle OSI ?",
    "Sur quelle couche travaille un routeur ?",
    "Que signifie DORA dans le DHCP ?",
    "Quel code HTTP indique qu'une ressource n'existe pas ?",
    "Quel est le prix d'un routeur Cisco ?",
]

if __name__ == "__main__":
    vector_store = ouvrir_vector_store()
    for question in QUESTIONS:
        print("=" * 70)
        print("QUESTION :", question)
        for doc, distance in vector_store.similarity_search_with_score(question, k=3):
            source = doc.metadata.get("source")
            page = doc.metadata.get("page")
            print(f"  [{distance:.3f}] {source} (page {page}) : {doc.page_content[:90]!r}")
