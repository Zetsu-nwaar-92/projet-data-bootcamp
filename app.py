import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle pré-entraîné
model = joblib.load('RegressionLogistique.pkl')

# Définir un dictionnaire de correspondance pour la variable 'region'
# Définir le mapping
region_mapping = {'FATICK': 2, 'DAKAR': 0, 'LOUGA': 7, 'TAMBACOUNDA': 11, 'KAOLACK': 4, 'THIES': 12, 'SAINT-LOUIS': 9,
                  'KOLDA': 6, 'KAFFRINE': 3, 'DIOURBEL': 1, 'ZIGUINCHOR': 13, 'MATAM': 8, 'SEDHIOU': 10, 'KEDOUGOU': 5}

# Définir un dictionnaire de correspondance pour la variable 'tenure'
# Définir le mapping
tenure_mapping = {'K > 24 month': 7, 'I 18-21 month': 5, 'H 15-18 month': 4, 'G 12-15 month': 3, 'J 21-24 month': 6,
                  'F 9-12 month': 2, 'E 6-9 month': 1, 'D 3-6 month': 0}

## Fonction pour effectuer les prédictions
def predict_subscription(region_mapping, tenure_mapping, region, tenure, freq_rech, freq, regularity):
    # Convertir les valeurs catégorielles en valeurs numériques
    region_code = region_mapping[region]
    tenure_code = tenure_mapping[tenure]  # Conversion de la variable 'tenure' en valeur numérique
    
    # Concaténer les valeurs numériques avec les autres caractéristiques
    features_with_categories = np.array([region_code, tenure_code, freq_rech, freq, regularity])
    
    # Effectuer la prédiction
    prediction = model.predict(features_with_categories.reshape(1, -1))
    return prediction

# Créer l'interface utilisateur avec Streamlit
st.title("Prédiction d'abonnement")

# Ajouter des champs pour les caractéristiques d'entrée
region = st.selectbox('Région', ['FATICK', 'DAKAR', 'THIES', 'SAINT-LOUIS', 'MATAM', 'LOUGA',
                                  'KOLDA', 'KAOLACK', 'DIOURBEL', 'DAKAR BANLIEUE', 'TAMBACOUNDA', 'SEDHIOU']) # Sélectionnez la région
tenure = st.selectbox('Ancienneté', ['K > 24 month', 'I 18-21 month', 'H 15-18 month', 'G 12-15 month',
                                      'J 21-24 month', 'F 9-12 month', 'E 6-9 month', 'D 3-6 month']) # Sélectionnez l'ancienneté
freq_rech = st.number_input('Fréquence de recharge') # Entrée pour la fréquence de recharge
freq = st.number_input('Fréquence') # Entrée pour la fréquence
regularity = st.number_input('Régularité') # Entrée pour la régularité

# Créer un bouton pour effectuer la prédiction lorsque l'utilisateur clique dessus
if st.button('Prédire'):
    # Créer un vecteur de caractéristiques à partir des entrées de l'utilisateur
    features = np.array([freq_rech, freq, regularity])
    # Effectuer la prédiction en passant également la régularité
    prediction = predict_subscription(region_mapping, tenure_mapping, region, tenure, freq_rech, freq, regularity)
    # Afficher le résultat de la prédiction
    if prediction == 1:
        st.write('Le client est susceptible de s\'abonner.')
    else:
        st.write('Le client n\'est pas susceptible de s\'abonner.')

