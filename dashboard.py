import streamlit as st
import json, pickle, time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(
    page_title="Supervision Réseau CHU",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def charger_modele():
    model = load_model("best_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("dataset_lstm_chu_v2.json") as f:
        data = json.load(f)
    return model, scaler, data["trajectoires"]

model, scaler, trajs = charger_modele()

LABELS = {
    0: "Normal",
    1: "Surcharge seule",
    2: "Harmoniques seules",
    3: "Sous-tension seule",
    4: "Mauvais cos phi",
    5: "Surcharge + Harmoniques",
    6: "Surcharge + Cos phi",
    7: "Critique mixte"
}

COULEURS = {
    0: "#2ecc71",
    1: "#e67e22",
    2: "#f1c40f",
    3: "#3498db",
    4: "#9b59b6",
    5: "#e74c3c",
    6: "#e74c3c",
    7: "#c0392b"
}

FEATURES = ["P_total_MW","Q_total_Mvar","S_total_MVA",
            "cos_phi","V_min_pu","Load_transfo_%",
            "Load_ligne_%","THD_%"]

if "historique" not in st.session_state:
    st.session_state.historique = []
if "idx_courant" not in st.session_state:
    st.session_state.idx_courant = np.random.randint(len(trajs))

st.markdown("## ⚡ Supervision Réseau Électrique — CHU")
st.markdown("---")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2,2,2])
with col_ctrl1:
    auto_mode = st.toggle("Simulation automatique", value=False)
with col_ctrl2:
    vitesse = st.slider("Vitesse (secondes)", 1, 5, 2)
with col_ctrl3:
    if st.button("Nouvelle trajectoire aléatoire"):
        st.session_state.idx_courant = np.random.randint(len(trajs))
        st.session_state.historique = []

st.markdown("---")

traj = trajs[st.session_state.idx_courant]
seq  = np.array(traj["sequence"])

seq_norm = scaler.transform(seq).reshape(1, 20, 8)
proba    = model.predict(seq_norm, verbose=0)[0]
pred     = int(np.argmax(proba))
reel     = int(traj["label_final"])
confiance = float(proba[pred]) * 100
couleur   = COULEURS[pred]

st.session_state.historique.append({
    "scenario": traj["scenario"],
    "pred": pred,
    "reel": reel,
    "confiance": confiance
})

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Scénario actuel", traj["scenario"])
with col2:
    st.metric("Classe réelle", LABELS[reel])
with col3:
    st.metric("Classe prédite", LABELS[pred])
with col4:
    st.metric("Confiance", f"{confiance:.1f}%")

st.markdown(f"""
<div style='background-color:{couleur}22;border-left:6px solid {couleur};
padding:16px;border-radius:8px;margin:16px 0'>
<h3 style='color:{couleur};margin:0'>
{("OK" if pred == reel else "ERREUR")} — {LABELS[pred]}
</h3>
</div>
""", unsafe_allow_html=True)

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Probabilités par classe")
    fig_bar = go.Figure(go.Bar(
        x=list(LABELS.values()),
        y=[float(p)*100 for p in proba],
        marker_color=[COULEURS[i] for i in range(8)],
        text=[f"{float(p)*100:.1f}%" for p in proba],
        textposition="outside"
    ))
    fig_bar.update_layout(
        yaxis_range=[0,110],
        height=350,
        margin=dict(t=20,b=20),
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_g2:
    st.subheader("Évolution des features sur 20 pas")
    feature_idx = st.selectbox("Feature à afficher", range(8),
                                format_func=lambda x: FEATURES[x])
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        y=seq[:, feature_idx],
        mode="lines+markers",
        line=dict(color=couleur, width=2),
        marker=dict(size=6)
    ))
    fig_line.update_layout(
        height=350,
        margin=dict(t=20,b=20),
        xaxis_title="Pas de temps",
        yaxis_title=FEATURES[feature_idx]
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.subheader("Valeurs des 8 features — moyenne sur la séquence")
moyennes = seq.mean(axis=0)
cols = st.columns(8)
for i, col in enumerate(cols):
    col.metric(FEATURES[i], f"{moyennes[i]:.3f}")

if len(st.session_state.historique) > 1:
    st.markdown("---")
    st.subheader("Historique des prédictions")
    hist = st.session_state.historique[-20:]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        y=[h["pred"] for h in hist],
        mode="lines+markers",
        name="Prédit",
        line=dict(color="#3498db", width=2),
        marker=dict(size=8)
    ))
    fig_hist.add_trace(go.Scatter(
        y=[h["reel"] for h in hist],
        mode="lines+markers",
        name="Réel",
        line=dict(color="#2ecc71", width=2, dash="dash"),
        marker=dict(size=8)
    ))
    fig_hist.update_layout(
        height=250,
        margin=dict(t=10,b=10),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(8)),
            ticktext=list(LABELS.values())
        )
    )
    st.plotly_chart(fig_hist, use_container_width=True)

if auto_mode:
    time.sleep(vitesse)
    st.session_state.idx_courant = np.random.randint(len(trajs))
    st.rerun()