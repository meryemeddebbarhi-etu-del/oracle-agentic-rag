# oracle-agentic-rag

Projet Oracle Agentic RAG Hybrid Search : un assistant qui répond à des questions
en cherchant dans nos propres documents, avec Oracle 26ai, LangGraph et Streamlit.

## Partie Base de données et documents (Membre 1)

Ce module prépare la base Oracle : connexion, lecture des documents,
découpage en morceaux, embeddings et indexation.

## Prérequis

- Python 3.11 ou plus (3.14 recommandé). Cocher « Add python.exe to PATH » à l'installation.
- Oracle Database 26ai (machine virtuelle ou Docker).
- Git.

## Installation

1. Cloner le dépôt et se placer dedans :
   ```
   git clone https://github.com/meryemeddebbarhi-etu-del/oracle-agentic-rag.git
   cd oracle-agentic-rag
   ```
2. Installer les bibliothèques :
   ```
   python -m pip install -r requirements.txt
   ```
3. Créer l'utilisateur Oracle. Se connecter en administrateur (sqlplus / as sysdba),
   se placer dans la base (ALTER SESSION SET CONTAINER = nom_de_votre_pdb;),
   puis exécuter une commande à la fois :
   ```sql
   CREATE USER vector IDENTIFIED BY vector DEFAULT TABLESPACE users QUOTA UNLIMITED ON users;
   GRANT CONNECT, RESOURCE TO vector;
   GRANT DB_DEVELOPER_ROLE TO vector;
   GRANT CREATE MINING MODEL TO vector;
   ```
4. Copier `.env.example` en `.env` et mettre vos propres valeurs :
   ```
   ORACLE_USER=vector
   ORACLE_PWD=vector
   ORACLE_DSN=ADRESSE_DE_VOTRE_ORACLE:1521/NOM_DU_SERVICE
   ```
   Le fichier `.env` ne doit jamais être publié sur GitHub.

## Tester la connexion

```
python database\test_connexion.py
```

Résultat attendu : `Connecté ! Oracle AI Database 26ai ...`

## Structure du dépôt

```
database/     connexion, ingestion et indexation (Membre 1)
documents/    documents de test (PDF, TXT)
.env.example  modèle de configuration
requirements.txt
```

## À venir

- Lecture et découpage des documents
- Embeddings et rangement dans Oracle
- Documents et questions de test