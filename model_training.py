# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import folium
import plotly.express as px
from geopy.distance import geodesic

# Étape 1 : Charger et préparer les données
print("Chargement des données agrégées...")
df = pd.read_csv('acled_burkina_aggregated.csv')

# Générer toutes les combinaisons de régions et semaines pour éviter les biais
regions = df['admin1'].unique()
years = range(2015, 2026)
weeks = range(1, 53)
all_combinations = [(region, year, week) for region in regions for year in years for week in weeks]
all_weeks = pd.DataFrame(all_combinations, columns=['admin1', 'year', 'week'])

# Fusionner avec les données existantes
df = all_weeks.merge(df, on=['admin1', 'year', 'week'], how='left')
df['attack_count'] = df['attack_count'].fillna(0)
df['fatalities'] = df['fatalities'].fillna(0)
df['latitude'] = df.groupby('admin1')['latitude'].transform(lambda x: x.fillna(x.mean()))
df['longitude'] = df.groupby('admin1')['longitude'].transform(lambda x: x.fillna(x.mean()))
df['target'] = (df['attack_count'] > 0).astype(int)

# Étape 2 : Créer toutes les features
# Features temporelles
df['month'] = df['week'].apply(lambda w: min(12, (w-1)//4 + 1))  # Approximation simple
df['is_rainy_season'] = df['month'].isin([6, 7, 8, 9]).astype(int)

# Features historiques (basées sur les données agrégées)
df = df.sort_values(by=['admin1', 'year', 'week'])
df['attack_count_prev_week'] = df.groupby('admin1')['attack_count'].shift(1).fillna(0)
df['fatalities_prev_week'] = df.groupby('admin1')['fatalities'].shift(1).fillna(0)
df['cumulative_attacks'] = df.groupby('admin1')['attack_count'].cumsum() - df['attack_count']

# Feature spatiale (encodage de la région)
le = LabelEncoder()
df['admin1_encoded'] = le.fit_transform(df['admin1'])

# Étape 3 : Diviser les données (approche temporelle)
train_df = df[(df['year'] >= 2015) & (df['year'] <= 2023)]
test_df = df[df['year'] >= 2024]

# Définir les features et la cible
features = ['month', 'is_rainy_season', 'latitude', 'longitude',
            'attack_count_prev_week', 'fatalities_prev_week', 'cumulative_attacks', 'admin1_encoded']
X_train = train_df[features].fillna(0)  # Remplir les valeurs manquantes par 0
y_train = train_df['target']
X_test = test_df[features].fillna(0)
y_test = test_df['target']

# Étape 4 : Gérer le déséquilibre avec SMOTE
print("Application de SMOTE pour équilibrer les classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Étape 5 : Entraîner le modèle XGBoost
print("Entraînement du modèle XGBoost...")
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_res, y_train_res)

# Étape 6 : Prédire et évaluer
print("Prédiction et évaluation...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
print("AUC-ROC :", roc_auc_score(y_test, y_pred_proba))

# Étape 7 : Visualiser les résultats
# Carte de risque basée sur les probabilités prédites
print("Création de la carte de risque...")
latest_week = test_df.groupby('admin1').last().reset_index()
latest_week['risk_proba'] = model.predict_proba(latest_week[features])[:, 1]

m = folium.Map(location=[12.2383, -1.5616], zoom_start=6)
for idx, row in latest_week.iterrows():
    color = 'red' if row['risk_proba'] > 0.7 else 'orange' if row['risk_proba'] > 0.3 else 'green'
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=10,
        popup=f"{row['admin1']} - Risque: {row['risk_proba']:.2f}",
        color=color,
        fill=True
    ).add_to(m)
m.save('burkina_risk_map.html')
print("Carte de risque enregistrée sous 'burkina_risk_map.html'.")

# Série temporelle des probabilités prédites (exemple sur les données de test)
test_df['predicted_proba'] = model.predict_proba(X_test)[:, 1]
fig = px.line(test_df, x='week', y='predicted_proba', color='admin1',
              title='Probabilités prédites par région (2024-2025)')
fig.write_html('burkina_predicted_proba.html')
print("Série temporelle enregistrée sous 'burkina_predicted_proba.html'.")

# Étape 8 : Enregistrer le modèle (optionnel)
import joblib
joblib.dump(model, 'xgb_model.pkl')
print("Modèle enregistré sous 'xgb_model.pkl'.")

print("\nModélisation terminée !")