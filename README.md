# 🌟 Prédiction de victoire de Pokémon avec l'IA

Ce projet est une application IA capable de prédire les chances de victoire d'un Pokémon en fonction de ses statistiques principales. Le modèle est entraîné avec **PyTorch**, et l'application déployée avec **Streamlit**.

---

## 📊 Objectif
Prédire si un Pokémon est **gagnant** ou **perdant** à partir des caractéristiques suivantes :

- HP
- Attack
- Defense
- Sp. Atk
- Sp. Def
- Speed
- Generation

---

## 🤖 Modèle utilisé
- ✔ PyTorch `nn.Module`
- Entraînement avec `BCEWithLogitsLoss` pour gérer le déséquilibre gagnant/perdant
- Optimiseur : Adam
- Normalisation avec `StandardScaler`

---

## 💡 Interface utilisateur
Une application Streamlit permet de :

- Saisir les statistiques d’un Pokémon
- Obtenir une prédiction de victoire
- Affichage d’un score de probabilité et d’un label clair

---

## 🔧 Utilisation locale

1. Clone le repo :
```bash
git clone https://github.com/ton-utilisateur/projet-pokemon-ia.git
cd projet-pokemon-ia
```

2. Installe les dépendances :
```bash
pip install -r requirements.txt
```

3. Lance l’app :
```bash
streamlit run app.py
```

---

## 🌐 Exemple de déploiement

- ✨ [App démo Streamlit](https://ton-lien-streamlit.app)

---

## 📚 Ce que j’ai appris
- Manipuler un dataset réel (Pokémon.csv)
- Créer un modèle binaire avec PyTorch
- Sauvegarder et recharger un modèle `.pt`
- Créer une app IA avec Streamlit
- Préparer un projet pour le déploiement

---

## 🚀 Auteur
Aurélie – Ingénieure en intelligence artificielle