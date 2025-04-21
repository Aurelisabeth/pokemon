import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Reconstruction directe du modèle (à l’identique de ton entraînement)
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

# Charger le modèle vide
model = PokemonClassifier()

# Charger les poids entraînés (sans classe custom)
model.load_state_dict(torch.load("model/model_pokemon.pt", map_location=torch.device("cpu")))
model.eval()

scaler = joblib.load("model/scaler_pokemon.pkl")

# 🎨 Streamlit config
st.set_page_config(page_title="Prédiction Pokémon", page_icon="🧬")
st.title("🧬 Prédiction IA – Peut-il gagner ?")
st.markdown("Choisis un Pokémon et vois si l’IA le juge comme **gagnant potentiel** en combat Pokémon !")

# 📄 Charger les données
df = pd.read_csv("data/Pokemon.csv")
liste_noms = df["Name"].sort_values().unique()

# 🔍 Choix du Pokémon
nom_pokemon = st.selectbox("🔎 Choisis ton Pokémon :", options=liste_noms)
donnees = df[df["Name"] == nom_pokemon].iloc[0]

# 🎛️ Sliders
st.sidebar.header("⚙️ Statistiques")
hp = st.sidebar.slider("HP (vie)", 1, 255, int(donnees["HP"]))
attack = st.sidebar.slider("Attaque", 1, 190, int(donnees["Attack"]))
defense = st.sidebar.slider("Défense", 1, 250, int(donnees["Defense"]))
sp_atk = st.sidebar.slider("Attaque Spéciale", 1, 194, int(donnees["Sp. Atk"]))
sp_def = st.sidebar.slider("Défense Spéciale", 1, 250, int(donnees["Sp. Defense"]))
speed = st.sidebar.slider("Vitesse", 1, 180, int(donnees["Speed"]))
generation = st.sidebar.slider("Génération", 1, 6, int(donnees["Generation"]))

# 📤 Prédiction
if st.button("🔮 Prédire avec l’IA"):
    X = pd.DataFrame([{
        "HP": hp,
        "Attack": attack,
        "Defense": defense,
        "Sp. Atk": sp_atk,
        "Sp. Defense": sp_def,
        "Speed": speed,
        "Generation": generation
    }])
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)
        proba = torch.sigmoid(output).item()
        gagnant = proba > 0.5

    st.subheader(f"🎯 Prédiction pour {nom_pokemon}")
    st.metric("Probabilité de victoire", f"{proba * 100:.2f} %")

    if gagnant:
        st.success("✨ Ce Pokémon a de grandes chances de GAGNER !")
    else:
        st.error("💀 Ce Pokémon a peu de chances de gagner...")
