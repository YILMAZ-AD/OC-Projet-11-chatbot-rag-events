"""
Script d'extraction et de pré-processing des données.
Ce script se connecte à l'API OpenAgenda, récupère les événements de Brest
sur la dernière année, nettoie les données et les sauvegarde au format JSON.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

def extract_and_clean_data():
    """
    Récupère, filtre et nettoie les événements culturels depuis l'API OpenAgenda.
    
    Étapes :
    1. Calcule la date d'il y a un an pour filtrer les événements récents.
    2. Gère la pagination de l'API pour récupérer tous les résultats.
    3. Nettoie le DataFrame (suppression des valeurs nulles et des doublons).
    4. Sauvegarde les données exploitables dans 'evenements_brest.json'.
    """
    
    # 1. Date dynamique : On calcule la date d'il y a exactement un an
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    print(f"Filtrage des événements à partir du : {one_year_ago}")

    headers = {
        "Content-Type": "application/json"
    }

    print("Lancement de la requête vers le portail public...")

    # 2. La boucle de pagination pour tout récupérer
    all_events = []
    offset = 0
    limit = 100

    while True:
        url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/?lang=fr&limit={limit}&offset={offset}&refine=location_city%3ABrest&where=firstdate_begin+%3E%3D+%27{one_year_ago}%27"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f" Erreur lors de la requête : {response.status_code}")
            break
            
        results = response.json().get('results', [])
        
        # S'il n'y a plus de résultats, on arrête la boucle
        if not results:
            break
            
        all_events.extend(results)
        offset += limit
        
        print(f" Récupération en cours... ({len(all_events)} événements trouvés)")
        
        # Si la page contient moins de 100 résultats, c'est qu'on a atteint la fin
        if len(results) < limit:
            break

    # Création du DataFrame avec tous les événements
    df = pd.DataFrame(all_events)

    if not df.empty:
        print(f"Succès total ! {len(df)} événements bruts récupérés à Brest.")
        
        # Sélection des colonnes utiles
        colonnes_utiles = [
            'uid', 'title_fr', 'description_fr', 'longdescription_fr', 
            'daterange_fr', 'firstdate_begin', 'location_name', 
            'location_address', 'conditions_fr', 'age_min', 'age_max','location_city'
        ]
        
        colonnes_presentes = [col for col in colonnes_utiles if col in df.columns]
        df_clean = df[colonnes_presentes]
        
        # 3. Le nettoyage (Gestion des données manquantes et dupliquées)
        # On supprime les lignes où il n'y a pas de titre OU pas de description
        df_clean = df_clean.dropna(subset=['title_fr', 'description_fr'])
        # On supprime les éventuels doublons basés sur l'ID unique
        df_clean = df_clean.drop_duplicates(subset='uid')
        
        print(f" Après nettoyage : {len(df_clean)} événements exploitables restants.")
        
        # Sauvegarde
        df_clean.to_json("evenements_brest.json", orient="records", force_ascii=False, indent=4)
        print(" le fichier JSON a été mis à jour avec succès !")
        
    else:
        print(" Aucun événement trouvé pour cette période.")

# Cette condition lance la fonction uniquement si on exécute ce script directement
if __name__ == "__main__":
    extract_and_clean_data()