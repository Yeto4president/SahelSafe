# Importation des bibliothèques nécessaires
import pandas as pd
import folium
import plotly.express as px
from geopy.distance import geodesic

# Étape 1 : Charger les données
print("Chargement des données...")
df = pd.read_csv('acled_burkina_2015_2025.csv')

# Étape 2 : Exploration initiale
print("Aperçu des données :")
print(df.head())
print("\nInformations sur les données :")
print(df.info())
print("\nNombre de valeurs manquantes :")
print(df.isnull().sum())

# Étape 3 : Nettoyage des données
# 3.1 Gérer les valeurs manquantes
print("\nGestion des valeurs manquantes...")
df['fatalities'] = df['fatalities'].fillna(0)  # Remplacer les valeurs manquantes dans 'fatalities' par 0
df = df.dropna(subset=['latitude', 'longitude'])  # Supprimer les lignes sans coordonnées

# 3.2 Formater les dates
print("Formatage des dates...")
df['event_date'] = pd.to_datetime(df['event_date'])
df['year'] = df['event_date'].dt.year
df['month'] = df['event_date'].dt.month
df['week'] = df['event_date'].dt.isocalendar().week

# 3.3 Filtrer les événements pertinents (déjà fait dans l'export ACLED, mais vérifions)
terrorism_events = ['Battles', 'Explosions/Remote violence', 'Violence against civilians']
df = df[df['event_type'].isin(terrorism_events)]
print("\nÉvénements filtrés :")
print(df['event_type'].value_counts())

# 3.4 Agréger les données par région et semaine
print("\nAgrégation des données...")
df_agg = df.groupby(['admin1', 'year', 'week']).agg({
    'fatalities': 'sum',
    'event_id_cnty': 'count',  # Nombre d'événements
    'latitude': 'mean',
    'longitude': 'mean'
}).reset_index()
df_agg.rename(columns={'event_id_cnty': 'attack_count'}, inplace=True)

# 3.5 Ajouter des features
# Feature 1 : Distance à l'attaque précédente
print("Ajout de la feature 'distance_to_prev'...")
df = df.sort_values(by=['event_date'])  # Trier par date pour que les attaques soient dans l'ordre
df['prev_lat'] = df['latitude'].shift(1)
df['prev_lon'] = df['longitude'].shift(1)
df['distance_to_prev'] = df.apply(
    lambda row: geodesic((row['latitude'], row['longitude']),
                         (row['prev_lat'], row['prev_lon'])).km
    if pd.notnull(row['prev_lat']) else 0, axis=1)

# Feature 2 : Saison des pluies (juin à septembre)
print("Ajout de la feature 'is_rainy_season'...")
df['is_rainy_season'] = df['month'].isin([6, 7, 8, 9]).astype(int)

# Étape 4 : Enregistrer les données nettoyées
print("\nEnregistrement des données nettoyées...")
df.to_csv('acled_burkina_cleaned.csv', index=False)
df_agg.to_csv('acled_burkina_aggregated.csv', index=False)

# Étape 5 : Visualisations
# 5.1 Carte des attaques
print("Création de la carte des attaques...")
m = folium.Map(location=[12.2383, -1.5616], zoom_start=6)  # Centrée sur le Burkina Faso
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['fatalities'] / 10,
        popup=f"{row['event_date']} - {row['event_type']}",
        color='red',
        fill=True
    ).add_to(m)
m.save('burkina_attacks_map.html')
print("Carte enregistrée sous 'burkina_attacks_map.html'.")

# 5.2 Série temporelle des attaques
print("Création de la série temporelle...")
df_time = df.groupby('event_date').size().reset_index(name='attack_count')
fig = px.line(df_time, x='event_date', y='attack_count', title='Attaques terroristes au Burkina Faso au fil du temps')
fig.write_html('burkina_attacks_timeseries.html')
print("Série temporelle enregistrée sous 'burkina_attacks_timeseries.html'.")

print("\nNettoyage et visualisation terminés !")