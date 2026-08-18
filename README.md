# Chatbot assistant avec un RAG intégré pour des recommandations d’événements culturels (Brest)

##  Contexte du projet

Trouver une idée de sortie sur internet, c'est souvent long. Il faut fouiller dans des agendas remplis d'événements qui ne nous intéressent pas forcément. 

Le projet **Puls-Events**, c'est un chatbot conçu pour simplifier l'accès à la culture dans la ville de Brest. L'idée est simple : au lieu de chercher soi-même sur un site web, on discute directement avec le bot. On lui dit ce qu'on veut, et il nous fait des recommandations précises.

##  Objectif Technique (Proof of Concept)

Le problème avec les IA classiques, c'est qu'elles inventent parfois des réponses quand elles ne savent pas (ce qu'on appelle les "hallucinations"). 

Pour éviter ça, ce projet utilise une technique appelée **RAG (Retrieval-Augmented Generation)**. En gros, le système fonctionne comme ça :
1. Le code va chercher les vrais événements à jour sur l'API publique **OpenAgenda**.
2. Il donne ces infos à l'IA (**Mistral AI**) avec une consigne stricte : elle n'a le droit de répondre qu'en utilisant ces événements-là, sans rien inventer d'autre.

##  Structure du projet

Le pipeline est divisé en 3 scripts principaux, à lancer dans cet ordre :

1. **`data_extract.py`** : Se connecte à l'API OpenAgenda, filtre les événements de Brest sur la dernière année, nettoie les données et sauvegarde le tout dans `evenements_brest.json`.
2. **`vectorize_data.py`** : Découpe les descriptions d'événements, les transforme en vecteurs mathématiques via Mistral, et crée une base de données locale dans le dossier `faiss_index/`.
3. **`chatbot.py`** : L'interface finale. Le script connecte la base de données et le modèle Mistral pour discuter avec l'utilisateur et lui faire des recommandations précises.

##  Comment lancer le projet ?

### 1. Préparation de l'environnement

Clonez ce dépôt, puis créez et activez un environnement virtuel Python :

```bash
python -m venv env_rag
# Sur Windows : env_rag\Scripts\activate
# Sur Mac/Linux : source env_rag/bin/activate
```

### 2. Installation des dépendances

Installez les bibliothèques requises :

```bash
pip install -r requirements.txt
```

### 3. Configuration de la clé API

Créez un fichier `.env` à la racine du projet et ajoutez-y votre clé Mistral AI :

```plaintext
MISTRAL_API_KEY=votre_cle_secrete_ici
```

### 4. Exécution du pipeline

Lancez les scripts un par un :

```bash
python data_extract.py
python vectorize_data.py
python chatbot.py
```
##  Tester et Évaluer le bot

Une fois que le projet est lancé, on peut vérifier que tout fonctionne bien et que le bot est fiable :

- **Faire un test rapide :** Vous pouvez lancer le script `test_pipeline.py` pour faire un essai et voir comment les données circulent dans le système de bout en bout.

- **Noter le bot avec Ragas :** J'ai aussi mis en place un système de correction automatique avec le script `evaluate_ragas.py`. En gros, l'outil **Ragas** agit comme un prof qui fait passer un examen au bot. Il va vérifier s'il n'a rien inventé (*Faithfulness*) et si sa réponse correspond bien à la question posée (*Answer Relevancy*). À la fin, ça sort un petit tableau de résultats pour prouver que le bot ne raconte pas n'importe quoi.

##  Stack Technique

Voici les technologies et bibliothèques utilisées pour construire ce projet :

- **Python** : Langage principal du projet.
- **LangChain** : Framework utilisée pour orchestrer le pipeline RAG (découpage des textes, gestion du prompt et connexion entre l'IA et la base de données).
- **FAISS** : Base de données vectorielle locale (développée par Meta) permettant une recherche de similarité ultra-rapide.
- **Mistral AI (API)** : Utilisation des modèles `mistral-embed` pour la vectorisation et `mistral-large-latest` pour la génération de texte (LLM).
- **Pandas & Requests** : Requêtage de l'API OpenAgenda, nettoyage et manipulation des données tabulaires (pre-processing).
- **python-dotenv** : Sécurisation des clés d'API hors du code source.

##  Auteur

**YAD95**  
Projet réalisé dans le cadre de la formation Data Engineer — OpenClassrooms
