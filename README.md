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

## Partie Recherche (Membre 2)

Ce module interroge la table `DOCUMENTS_RAG` déjà créée et remplie par
Membre 1 (connexion, index vectoriel HNSW/IVF, index Oracle Text) et propose
trois méthodes de recherche, chacune renvoyant une liste de `SearchResult` au
format identique :

| Méthode | Fichier | Principe |
|---|---|---|
| `vector_search(query, top_k)` | `search/vector_search.py` | Similarité cosinus sur `embedding` (`VECTOR_DISTANCE`), utilise l'index HNSW/IVF de Membre 1 |
| `keyword_search(query, top_k)` | `search/keyword_search.py` | `CONTAINS(text, ...)` sur l'index Oracle Text `DOCUMENTS_RAG_TEXT_IDX` de Membre 1 |
| `hybrid_search(query, top_k)` | `search/hybrid_search.py` | Fusion des deux listes ci-dessus par **Reciprocal Rank Fusion (RRF)** |

Aucun de ces modules ne crée de table ni d'index : ils réutilisent entièrement
ce que `database/ingestion.py` a déjà mis en place.

### Pourquoi RRF pour la recherche hybride ?

Le score vectoriel (similarité cosinus, environ -1 à 1) et le score Oracle
Text (`SCORE(1)`, un entier non borné propre à Oracle Text) ne sont pas sur la
même échelle. RRF ne combine que le **rang** de chaque document dans chaque
liste (1er, 2e, 3e...), pas la valeur brute du score, ce qui évite toute
normalisation arbitraire. Détails, formule et limites assumées : voir les
commentaires en tête de `search/hybrid_search.py`.

### Format de résultat (interface pour Membre 3)

```python
from search import vector_search, keyword_search, hybrid_search, SearchError

resultats = hybrid_search("comment fonctionne le DHCP ?", top_k=5)
for r in resultats:
    print(r.chunk_id, r.document_id, r.source, r.score, r.search_method)
    print(r.content[:200])
```

Chaque `SearchResult` expose : `chunk_id`, `document_id`, `content`, `source`,
`score`, `search_method`, `metadata` (dict). `document_id` est le nom du
fichier source (`metadata["source"]`) : le schéma de Membre 1 ne contient pas
d'identifiant de document distinct du nom de fichier.

Les trois fonctions lèvent `search.SearchError` (jamais une exception Oracle
brute) en cas de requête vide, de `top_k` invalide, d'échec de connexion, ou
d'index manquant — avec un message explicite indiquant quoi corriger.

### Exécuter les tests

```
python -m pytest tests/ -v
```

- `tests/test_search_utils.py`, `tests/test_hybrid_fusion.py` : tests **purs**
  (validation, construction de requête Oracle Text, fusion RRF avec des
  doubles de test) — s'exécutent partout, aucune base Oracle requise.
- `tests/test_vector_search.py`, `tests/test_keyword_search.py`,
  `tests/test_hybrid_search.py` : tests **d'intégration**, nécessitent la base
  Oracle de Membre 1 déjà remplie (`python database\ingestion.py` exécuté
  avant) et un accès réseau pour le modèle d'embeddings. Si la base n'est pas
  joignable, ils sont **SKIP** (pas échoués) avec un message explicite.

## Structure du dépôt

```
database/     connexion, ingestion et indexation (Membre 1)
search/       vector_search, keyword_search, hybrid_search, résultat standardisé (Membre 2)
tests/        tests unitaires (purs) et tests d'intégration (base Oracle requise)
documents/    documents de test (PDF, TXT)
.env.example  modèle de configuration
requirements.txt
```

## À venir

- Agent LangGraph / ReAct s'appuyant sur `search.hybrid_search` (Membre 3)
- Interface Streamlit