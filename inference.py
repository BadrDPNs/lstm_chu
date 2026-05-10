import json
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model  = load_model("best_model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

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

with open("dataset_lstm_chu_v2.json") as f:
    data = json.load(f)

trajs = data["trajectoires"]

print("=" * 55)
print("   TEST D'INFÉRENCE — RÉSEAU ÉLECTRIQUE CHU")
print("=" * 55)

indices = np.random.choice(len(trajs), 5, replace=False)

for i in indices:
    t = trajs[i]
    seq = np.array(t["sequence"])

    seq_norm = scaler.transform(seq).reshape(1, 20, 8)
    proba    = model.predict(seq_norm, verbose=0)[0]
    pred     = int(np.argmax(proba))
    reel     = int(t["label_final"])
    confiance = float(proba[pred]) * 100

    statut = "✓ OK" if pred == reel else "✗ ERREUR"

    print(f"\nTrajectoire #{t['traj_id']} | Scénario : {t['scenario']}")
    print(f"  Réel      : {LABELS[reel]}")
    print(f"  Prédit    : {LABELS[pred]}  ({confiance:.1f}% confiance)")
    print(f"  Résultat  : {statut}")
    print(f"  Top 3 probabilités :")
    top3 = np.argsort(proba)[::-1][:3]
    for idx in top3:
        print(f"    - {LABELS[idx]:<25} {proba[idx]*100:.1f}%")

print("\n" + "=" * 55)
print("Inférence terminée.")