"""
Script de vectorisation et de création de l'index FAISS.
Ce script charge les événements nettoyés, les découpe en morceaux intelligents (chunking),
les transforme en vecteurs via Mistral AI, et sauvegarde l'index localement.
"""

import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chargement des variables d'environnement (clé API)
load_dotenv()

def create_and_save_vector_store():
    """
    Charge le JSON, crée les chunks avec métadonnées, génère les embeddings 
    et sauvegarde la base vectorielle FAISS.
    
    Retourne :
        vector_store (FAISS) : L'objet FAISS contenant l'index vectoriel.
    """
    print(" Chargement des événements depuis le fichier JSON...")
    with open("evenements_brest.json", "r", encoding="utf-8") as f:
        events = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []

    for event in events:
        title = event.get('title_fr', '')
        lieu = event.get('location_name', '')
        adresse = event.get('location_address', '')
        date = event.get('daterange_fr', '')
        conditions = event.get('conditions_fr')
        age_min = event.get('age_min')
        age_max = event.get('age_max')
        description = event.get('longdescription_fr') or event.get('description_fr', '')

        # En-tête toujours présent, quel que soit le chunk
        header = f"Titre: {title}\nLieu: {lieu} - {adresse}\nDate: {date}\n"
        if conditions:
            header += f"Conditions et prix: {conditions}\n"
        if age_min is not None or age_max is not None:
            header += f"Public visé: de {age_min if age_min is not None else '0'} à {age_max if age_max is not None else '99'} ans\n"

        metadata = {
            "uid": event.get("uid"),
            "title": title,
            "lieu": lieu,
            "adresse": adresse,
            "date_lisible": date,
            "date_informatique": event.get("firstdate_begin"),
        }

        # On ne découpe QUE la description
        desc_chunks = text_splitter.split_text(description) if description else [""]
        for i, chunk in enumerate(desc_chunks):
            content = header + f"Description: {chunk}"
            documents.append(Document(page_content=content, metadata={**metadata, "chunk_index": i}))

    print(f" {len(events)} événements → {len(documents)} chunks (en-tête préservé dans chacun).")

    # Vectorisation + indexation FAISS
    print(" Envoi des chunks à Mistral pour vectorisation...")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = FAISS.from_documents(documents, embeddings)

    # Vérification : tous les événements sont bien indexés
    print(f" {vector_store.index.ntotal} vecteurs dans l'index.")

    vector_store.save_local("faiss_index")
    print(" Base vectorielle sauvegardée dans 'faiss_index' !")
    
    return vector_store


def search_events(vector_store, query, k=3, fetch_k=10):
    """
    Recherche en excluant les chunks en double d'un même événement.
    
    Arguments :
        vector_store : L'index FAISS dans lequel chercher.
        query (str) : La question ou mots-clés de l'utilisateur.
        k (int) : Nombre de résultats uniques finaux voulus.
        fetch_k (int) : Nombre de résultats bruts à remonter avant déduplication.
        
    Retourne :
        unique_results (list) : Liste des documents (événements) uniques trouvés.
    """
    results = vector_store.similarity_search(query, k=fetch_k)
    seen_uids = set()
    unique_results = []
    
    for doc in results:
        uid = doc.metadata.get("uid")
        if uid not in seen_uids:
            seen_uids.add(uid)
            unique_results.append(doc)
        # On s'arrête dès qu'on a 'k' événements uniques
        if len(unique_results) == k:
            break
            
    return unique_results


# Bloc principal d'exécution
if __name__ == "__main__":
    # 1. On lance la création de l'index
    mon_vector_store = create_and_save_vector_store()
    
    # 2. On lance un test rapide de vérification
    print("\n Test pour : 'concert de musique gratuit'")
    test_results = search_events(mon_vector_store, "concert de musique gratuit", k=3)
    
    for i, doc in enumerate(test_results, 1):
        print(f"{i}. {doc.metadata.get('title')} — {doc.metadata.get('date_lisible')} ({doc.metadata.get('lieu')})")