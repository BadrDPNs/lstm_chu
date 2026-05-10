import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("=== Chargement des données ===")
with open("dataset_lstm_chu_v2.json") as f:
    data = json.load(f)

trajs = data["trajectoires"]
X = np.array([t["sequence"]    for t in trajs])
y = np.array([t["label_final"] for t in trajs])
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")

print("\n=== Split train / val / test ===")
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
)
print(f"Train : {X_train.shape[0]} | Val : {X_val.shape[0]} | Test : {X_test.shape[0]}")

print("\n=== Normalisation ===")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.reshape(-1, 8)).reshape(-1, 20, 8)
X_val_s   = scaler.transform(X_val.reshape(-1, 8)).reshape(-1, 20, 8)
X_test_s  = scaler.transform(X_test.reshape(-1, 8)).reshape(-1, 20, 8)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler sauvegardé.")

print("\n=== Encodage labels ===")
y_train_oh = to_categorical(y_train, num_classes=8)
y_val_oh   = to_categorical(y_val,   num_classes=8)
y_test_oh  = to_categorical(y_test,  num_classes=8)

classes = np.arange(8)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(enumerate(weights))
print("Poids des classes :", {k: round(v, 2) for k, v in class_weight_dict.items()})

print("\n=== Construction du modèle LSTM ===")
model = Sequential([
    Input(shape=(20, 8)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(8,  activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("\n=== Entraînement ===")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
]

history = model.fit(
    X_train_s, y_train_oh,
    validation_data=(X_val_s, y_val_oh),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n=== Évaluation sur le test set ===")
y_pred = np.argmax(model.predict(X_test_s), axis=1)
y_true = np.argmax(y_test_oh, axis=1)

LABELS = ["Normal","Surcharge","Harmoniques","Sous-tension",
          "Cos phi","S+Harm","S+CosPhi","Critique"]
print(classification_report(y_true, y_pred, target_names=LABELS))

print("\n=== Sauvegarde des graphiques ===")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS)
plt.title("Matrice de confusion")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("confusion_matrix.png sauvegardé.")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"],     label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"],     label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy"); plt.legend()
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
print("learning_curves.png sauvegardé.")

print("\n=== Terminé ! ===")