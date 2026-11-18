"""Compare vector / keyword / hybrid sur le corpus Oracle + PL/SQL et aide a
calibrer SEUIL_NO_ANSWER_VECTOR.

Usage : python comparer_methodes.py
(Oracle doit etre demarre et la table DOCUMENTS_RAG remplie.)

Pour chaque question on affiche, par methode : le temps, le meilleur score et
si le fichier attendu apparait dans le top 3. A la fin, le script propose un
seuil entre le pire score des questions pertinentes et le meilleur score des
questions hors sujet.
"""

import sys
import time

from search import hybrid_search, keyword_search, vector_search

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TOP_K = 5

# (type, question, fragment(s) du nom de fichier attendu)
#   "pertinente" : la reponse est dans le corpus
#   "hors_sujet" : aucune reponse attendue
#   "proche"     : sujet voisin, peu ou pas traite (sert a voir le comportement)
QUESTIONS = [
    ("pertinente", "Quelle est la différence entre une procédure et une fonction ?",
     ("fonctions_stockees", "procedures_stockees")),
    ("pertinente", "À quoi sert la clause WHERE CURRENT OF ?", ("curseurs",)),
    ("pertinente", "Quels sont les modes SHUTDOWN d'Oracle ?", ("demarrage_arret",)),
    ("pertinente", "Quel est le port par défaut du listener ?", ("net_listener",)),
    ("pertinente", "Que signifie WITH ADMIN OPTION ?", ("utilisateurs_privileges",)),
    ("pertinente", "Comment exporter un schéma avec expdp ?", ("datapump",)),
    ("pertinente", "Qu'est-ce que la création différée des segments ?", ("stockage_tablespaces",)),
    ("pertinente", "Comment exécuter une instruction SQL dynamique ?", ("sql_dynamique",)),
    ("pertinente", "Que se passe-t-il si SELECT INTO ne retourne aucune ligne ?",
     ("blocs_declarations", "exceptions")),
    ("pertinente", "Comment créer une PDB à partir du seed ?", ("pdb_creation",)),
    ("hors_sujet", "Quelle est la recette de la tarte aux pommes ?", ()),
    ("hors_sujet", "Qui a gagné la Coupe du monde de football en 2018 ?", ()),
    ("hors_sujet", "Comment faire pousser des tomates sur un balcon ?", ()),
    ("proche", "Comment faire une sauvegarde avec RMAN ?", ()),
    ("proche", "Comment configurer la réplication MySQL ?", ()),
]

METHODES = (
    ("vector", vector_search),
    ("keyword", keyword_search),
    ("hybrid", hybrid_search),
)


def attendu_dans(resultats, fragments):
    """True si l'un des 3 premiers resultats vient d'un fichier attendu."""
    for r in resultats[:3]:
        nom = (r.document_id or "").lower()
        if any(f in nom for f in fragments):
            return True
    return False


def main():
    # Echauffement : le premier appel charge le modele d'embeddings (~30 s)
    print("Chargement du modele d'embeddings (echauffement)...")
    vector_search("echauffement", top_k=1)
    print()

    meilleurs_pertinents = []
    meilleurs_hors_sujet = []
    reussites = {m: 0 for m, _ in METHODES}
    nb_pertinentes = 0

    for type_q, question, fragments in QUESTIONS:
        print(f"=== [{type_q}] {question}")
        top1_vector = None
        for nom, fonction in METHODES:
            debut = time.perf_counter()
            try:
                resultats = fonction(question, top_k=TOP_K)
            except Exception as erreur:  # on continue meme si une methode echoue
                print(f"  {nom:8} ERREUR : {erreur}")
                continue
            ms = (time.perf_counter() - debut) * 1000
            if not resultats:
                print(f"  {nom:8} {ms:5.0f} ms | aucun resultat")
                continue

            top = resultats[0]
            if nom == "vector":
                top1_vector = top.score
            marque = ""
            if type_q == "pertinente":
                ok = attendu_dans(resultats, fragments)
                marque = " | attendu dans le top 3 : " + ("OUI" if ok else "NON")
                if ok:
                    reussites[nom] += 1
            print(
                f"  {nom:8} {ms:5.0f} ms | top1 = {top.document_id} "
                f"(score {top.score:.4f}){marque}"
            )

        if type_q == "pertinente":
            nb_pertinentes += 1
            if top1_vector is not None:
                meilleurs_pertinents.append(top1_vector)
        elif type_q == "hors_sujet" and top1_vector is not None:
            meilleurs_hors_sujet.append(top1_vector)
        print()

    print("=" * 70)
    print("BILAN")
    for nom, _ in METHODES:
        print(f"  {nom:8} : fichier attendu dans le top 3 pour {reussites[nom]}/{nb_pertinentes} questions pertinentes")

    if meilleurs_pertinents and meilleurs_hors_sujet:
        bas_pertinent = min(meilleurs_pertinents)
        haut_hors_sujet = max(meilleurs_hors_sujet)
        print()
        print(f"  Score vectoriel top 1, questions pertinentes : min = {bas_pertinent:.4f}")
        print(f"  Score vectoriel top 1, questions hors sujet  : max = {haut_hors_sujet:.4f}")
        if haut_hors_sujet < bas_pertinent:
            seuil = (bas_pertinent + haut_hors_sujet) / 2
            print(f"  -> SEUIL_NO_ANSWER_VECTOR conseille : {seuil:.2f}")
        else:
            print("  -> Chevauchement : aucun seuil ne separe proprement les deux groupes.")
            print("     Regarde les questions pertinentes avec un score bas et reformule-les.")


if __name__ == "__main__":
    main()