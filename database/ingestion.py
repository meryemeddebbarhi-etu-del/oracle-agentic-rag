import os
import re
from pathlib import Path

import oracledb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.oraclevs import DistanceStrategy, create_index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DOSSIER_DOCS = Path(__file__).resolve().parent.parent / "documents"

TAILLE_MORCEAU = 500   # nombre de caractères par morceau
CHEVAUCHEMENT = 80     # caractères répétés entre deux morceaux

MODELE_EMBEDDING = "sentence-transformers/all-MiniLM-L12-v2"
NOM_TABLE = "DOCUMENTS_RAG"
INDEX_VECTORIEL = "DOCUMENTS_RAG_HNSW_IDX"
INDEX_IVF = "DOCUMENTS_RAG_IVF_IDX"
INDEX_TEXTE = "DOCUMENTS_RAG_TEXT_IDX"


def lire_fichier(chemin: Path) -> list[dict]:
    """Lit un PDF ou un TXT et renvoie une liste de textes avec leurs étiquettes."""
    if chemin.suffix.lower() == ".pdf":
        pages = []
        for numero, page in enumerate(PdfReader(chemin).pages, start=1):
            texte = page.extract_text() or ""
            if texte.strip():
                pages.append({"texte": texte, "source": chemin.name, "page": numero})
        return pages

    if chemin.suffix.lower() == ".txt":
        texte = chemin.read_text(encoding="utf-8", errors="ignore")
        return [{"texte": texte, "source": chemin.name, "page": 1}]

    return []


def lire_tous_les_documents() -> list[dict]:
    resultats = []
    for chemin in sorted(DOSSIER_DOCS.iterdir()):
        resultats.extend(lire_fichier(chemin))
    return resultats


def nettoyer_texte(texte: str) -> str:
    """Enlève les espaces et sauts de ligne en trop."""
    texte = texte.replace("\r", "")
    texte = re.sub(r"[ \t]+", " ", texte)      # espaces multiples -> un seul
    texte = re.sub(r"\n{3,}", "\n\n", texte)   # plus de 2 sauts de ligne -> 2
    return texte.strip()


def decouper_documents(documents: list[dict]) -> list[dict]:
    """Découpe chaque texte en morceaux et ajoute les étiquettes."""
    decoupeur = RecursiveCharacterTextSplitter(
        chunk_size=TAILLE_MORCEAU,
        chunk_overlap=CHEVAUCHEMENT,
    )
    morceaux = []
    for doc in documents:
        for texte in decoupeur.split_text(nettoyer_texte(doc["texte"])):
            morceaux.append({
                "texte": texte,
                "source": doc["source"],
                "page": doc["page"],
                "categorie": "general",
                "chunk_id": len(morceaux) + 1,
            })
    return morceaux


def creer_index(conn, vector_store) -> None:
    """Crée l'index vectoriel (HNSW, sinon IVF) et l'index texte (mots-clés).

    Une erreur sur un index n'arrête pas le programme : on affiche un message.
    """
    try:
        create_index(
            conn,
            vector_store,
            params={
                "idx_name": INDEX_VECTORIEL,
                "idx_type": "HNSW",
                "accuracy": 95,
                "parallel": 4,
            },
        )
        print(f"Index vectoriel HNSW créé : {INDEX_VECTORIEL}")
    except Exception as erreur:
        print(f"Index HNSW NON créé ({erreur}). Essai avec un index IVF...")
        try:
            cur = conn.cursor()
            cur.execute(
                f"CREATE VECTOR INDEX {INDEX_IVF} ON {NOM_TABLE}(embedding) "
                "ORGANIZATION NEIGHBOR PARTITIONS "
                "DISTANCE COSINE WITH TARGET ACCURACY 90"
            )
            conn.commit()
            cur.close()
            print(f"Index vectoriel IVF créé : {INDEX_IVF}")
        except Exception as erreur_ivf:
            print(f"Index vectoriel IVF NON créé : {erreur_ivf}")

    try:
        cur = conn.cursor()
        cur.execute(
            f"CREATE INDEX {INDEX_TEXTE} ON {NOM_TABLE}(text) "
            "INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('SYNC (ON COMMIT)')"
        )
        conn.commit()
        cur.close()
        print(f"Index texte créé : {INDEX_TEXTE}")
    except Exception as erreur:
        print(f"Index texte NON créé : {erreur}")


def ranger_dans_oracle(morceaux: list[dict]) -> int:
    """Calcule les embeddings, range les morceaux dans Oracle et crée les index."""
    load_dotenv()
    conn = oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PWD"),
        dsn=os.getenv("ORACLE_DSN"),
    )

    # Repartir d'une table propre à chaque lancement (les index partent avec elle)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM user_tables WHERE table_name = :1", [NOM_TABLE])
    if cur.fetchone()[0] > 0:
        cur.execute(f"DROP TABLE {NOM_TABLE} PURGE")
        conn.commit()

    documents = [
        Document(
            page_content=m["texte"],
            metadata={
                "source": m["source"],
                "page": m["page"],
                "categorie": m["categorie"],
                "chunk_id": m["chunk_id"],
            },
        )
        for m in morceaux
    ]

    embeddings = HuggingFaceEmbeddings(model_name=MODELE_EMBEDDING)
    vector_store = OracleVS.from_documents(
        documents,
        embeddings,
        client=conn,
        table_name=NOM_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
    )
    conn.commit()

    cur.execute(f"SELECT COUNT(*) FROM {NOM_TABLE}")
    nombre = cur.fetchone()[0]
    cur.close()

    creer_index(conn, vector_store)

    conn.close()
    return nombre


if __name__ == "__main__":
    documents = lire_tous_les_documents()
    morceaux = decouper_documents(documents)
    print(f"{len(documents)} texte(s) lu(s), {len(morceaux)} morceau(x) créé(s).")

    print("Calcul des embeddings et rangement dans Oracle...")
    nombre = ranger_dans_oracle(morceaux)
    print(f"Terminé : {nombre} morceau(x) rangé(s) dans la table {NOM_TABLE}.")