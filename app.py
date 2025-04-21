import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import joblib

import torch
import torch.nn as nn

# Classe utilisée à l'entraînement
class PokemonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Chargement du modèle
model = PokemonClassifier()
model.load_state_dict(torch.load("model/pokemon_state_dict.pt"))
model.eval()

# Chargement du scaler
scaler = joblib.load("scaler_pokemon.pkl")

# Titre de l'app
st.title("🔮 Prédiction de victoire d’un Pokémon")

st.markdown("""
Remplis les caractéristiques d’un Pokémon pour savoir s’il a **de bonnes chances de gagner** !  
*(Modèle entraîné sur un dataset de combats entre Pokémon)*
""")

# Entrées utilisateur
col1, col2 = st.columns(2)
with col1:
    hp = st.slider("HP", 0, 255, 60)
    attack = st.slider("Attack", 0, 190, 70)
    defense = st.slider("Defense", 0, 250, 70)
    speed = st.slider("Speed", 0, 200, 65)
with col2:
    sp_atk = st.slider("Sp. Attack", 0, 200, 70)
    sp_def = st.slider("Sp. Defense", 0, 250, 70)
    generation = st.selectbox("Génération", [1, 2, 3, 4, 5, 6])

# Préparation des données
input_data = pd.DataFrame([{
    "HP": hp,
    "Attack": attack,
    "Defense": defense,
    "Sp. Atk": sp_atk,
    "Sp. Def": sp_def,
    "Speed": speed,
    "Generation": generation
}])

scaled = scaler.transform(input_data)
tensor_input = torch.tensor(scaled, dtype=torch.float32)

# Prédiction
with torch.no_grad():
    logits = model(tensor_input)
    proba = torch.sigmoid(logits).item()

    st.subheader("📊 Résultat")
    st.write(f"Probabilité de victoire : **{proba * 100:.2f} %**")

    if proba > 0.5:
        st.success("🎉 Ce Pokémon a de **grandes chances de gagner !**")
    else:
        st.warning("🤔 Ce Pokémon a **peu de chances de gagner.**")
