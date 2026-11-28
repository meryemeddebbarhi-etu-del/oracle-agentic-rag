"""Connexion Oracle réutilisable, en lecture seule, pour les Membres 2, 3 et 4.

Ce module n'exécute AUCUN DROP TABLE et ne relance pas l'ingestion.
Il se branche sur la table DOCUMENTS_RAG déjà remplie par ingestion.py.

Exemple d'utilisation :

    from database.connexion_lecture import ouvrir_vector_store

    vector_store = ouvrir_vector_store()
    resultats = vector_store.similarity_search_with_score("Qu'est-ce que TCP ?", k=3)
"""

import os

import oracledb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.oraclevs import DistanceStrategy

NOM_TABLE = "DOCUMENTS_RAG"
# Même modèle que pour l'ingestion : obligatoire pour comparer les vecteurs.
MODELE_EMBEDDING = "sentence-transformers/all-MiniLM-L12-v2"


def ouvrir_connexion():
    """Ouvre une connexion Oracle avec les valeurs du fichier .env."""
    load_dotenv()
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PWD"),
        dsn=os.getenv("ORACLE_DSN"),
    )


def ouvrir_embeddings():
    """Charge le modèle d'embedding utilisé pour l'ingestion."""
    return HuggingFaceEmbeddings(model_name=MODELE_EMBEDDING)


def ouvrir_vector_store(conn=None):
    """Renvoie un OracleVS branché sur la table DOCUMENTS_RAG existante.

    Si la table existe déjà (cas normal), elle n'est ni supprimée ni modifiée.
    """
    if conn is None:
        conn = ouvrir_connexion()
    return OracleVS(
        client=conn,
        embedding_function=ouvrir_embeddings(),
        table_name=NOM_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
    )
