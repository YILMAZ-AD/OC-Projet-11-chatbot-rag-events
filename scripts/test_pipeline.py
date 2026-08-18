"""
Script de tests unitaires pour le pipeline RAG (Puls-Events).
Utilisation : lancer la commande `pytest test_pipeline.py -v` dans le terminal.
"""

import os
import json
import pytest
from datetime import datetime, timedelta
from langchain_core.documents import Document
from chatbot import format_events

def test_data_file_exists():
    """
    Test 1 : Vérifie que le script d'extraction a bien généré le fichier JSON.
    """
    assert os.path.exists("evenements_brest.json"), " Le fichier evenements_brest.json est introuvable."

def test_faiss_index_exists():
    """
    Test 2 : Vérifie que le dossier de la base vectorielle FAISS a bien été créé.
    """
    assert os.path.exists("faiss_index"), " Le dossier faiss_index est introuvable."
    assert os.path.exists(os.path.join("faiss_index", "index.faiss")), " Le fichier index.faiss est manquant à l'intérieur du dossier."

def test_format_events():
    """
    Test 3 : Test unitaire pur sur la fonction de formatage du RAG.
    """
    doc1 = Document(page_content="Titre: Concert de Jazz\nLieu: Le Quartz")
    doc2 = Document(page_content="Titre: Expo Peinture\nLieu: Musée des Beaux-Arts")
    docs = [doc1, doc2]
    
    result = format_events(docs)
    expected_result = "Titre: Concert de Jazz\nLieu: Le Quartz\n\n---\n\nTitre: Expo Peinture\nLieu: Musée des Beaux-Arts"
    
    assert result == expected_result, " Le formatage des événements est incorrect."

def test_events_are_recent_and_in_brest():
    """
    Test 4 : Vérifie que TOUS les événements du JSON respectent les critères
    métier fixés par Jérémy : moins d'un an d'ancienneté et localisés à Brest.
    """
    with open("evenements_brest.json", "r", encoding="utf-8") as f:
        events = json.load(f)

    assert len(events) > 0, " Le fichier JSON est vide."

    one_year_ago = datetime.now() - timedelta(days=365)

    for event in events:
        # Critère 1 : l'événement ne doit pas avoir plus d'un an
        # On utilise [:10] pour s'assurer de ne prendre que l'année, le mois et le jour (YYYY-MM-DD)
        date_str = event["firstdate_begin"][:10]
        date_event = datetime.strptime(date_str, "%Y-%m-%d")
        assert date_event >= one_year_ago, (
            f" Événement trop ancien : {event.get('title_fr')} ({event['firstdate_begin']})"
        )

        # Critère 2 : l'événement doit être localisé à Brest
        assert event.get("location_city") == "Brest", (
            f" Événement hors zone : {event.get('title_fr')} ({event.get('location_city')})"
        )