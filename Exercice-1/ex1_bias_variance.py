"""
Exercice 1: Analyse Bias/Variance avec Keras
TP2 Deep Learning Engineering - ENSPY 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# Configuration pour reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("EXERCICE 1 : ANALYSE BIAS/VARIANCE")
print("=" * 60)

# 1. Chargement des données MNIST
print("\n[1] Chargement des donnees MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"   - Donnees initiales : {x_train.shape[0]} images d'entrainement")
print(f"   - Donnees de test : {x_test.shape[0]} images")

# 2. Création des ensembles Training / Validation / Test
print("\n[2] Division en Training (90%) / Validation (10%)...")

# Utiliser 90% pour training et 10% pour validation
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]

print(f"   - Training set : {x_train.shape[0]} images")
print(f"   - Validation set : {x_val.shape[0]} images")
print(f"   - Test set : {x_test.shape[0]} images")

# 3. Normalisation et reshaping
print("\n[3] Normalisation et pretraitement...")
x_train = x_train.reshape(54000, 784).astype("float32") / 255.0
x_val = x_val.reshape(6000, 784).astype("float32") / 255.0
x_test = x_test.reshape(10000, 784).astype("float32") / 255.0

# Conversion des labels en one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print("   - Donnees normalisees (0-1)")
print("   - Images aplaties (28x28 -> 784)")
print("   - Labels en one-hot encoding (10 classes)")

# 4. Construction du modèle
print("\n[4] Construction du modele...")
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n   Architecture du modele :")
model.summary()

# 5. Entraînement avec validation
print("\n[5] Entrainement du modele (5 epoques)...")
print("   Observation : Training vs Validation performance")
print("-" * 60)

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

# 6. Évaluation sur l'ensemble de test
print("\n[6] Evaluation finale sur le Test Set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"   Test Loss : {test_loss:.4f}")
print(f"   Test Accuracy : {test_accuracy * 100:.2f}%")

# 7. Analyse des résultats
print("\n" + "=" * 60)
print("ANALYSE BIAS/VARIANCE")
print("=" * 60)

final_train_loss = history.history['loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\nResultats finaux (Epoque 5) :")
print(f"   Training Loss     : {final_train_loss:.4f}")
print(f"   Training Accuracy : {final_train_acc * 100:.2f}%")
print(f"   Validation Loss   : {final_val_loss:.4f}")
print(f"   Validation Accuracy : {final_val_acc * 100:.2f}%")
print(f"   Test Accuracy     : {test_accuracy * 100:.2f}%")

# Diagnostic
print("\n[DIAGNOSTIC] :")
train_val_gap = (final_train_acc - final_val_acc) * 100

diagnostic = ""
if final_train_acc < 0.90 and final_val_acc < 0.90:
    diagnostic = "HIGH BIAS (Underfitting)"
    print(f"   {diagnostic}")
    print("   -> Le modele performe mal sur training ET validation")
    print("   -> Solution : Augmenter la complexite du modele")
elif train_val_gap > 5:
    diagnostic = "HIGH VARIANCE (Overfitting)"
    print(f"   {diagnostic}")
    print(f"   -> Ecart train-val : {train_val_gap:.2f}%")
    print("   -> Le modele surappprend les donnees d'entrainement")
    print("   -> Solution : Regularisation (L2, Dropout, Data augmentation)")
else:
    diagnostic = "GOOD FIT"
    print(f"   {diagnostic}")
    print(f"   -> Ecart train-val : {train_val_gap:.2f}%")
    print("   -> Le modele generalise bien")

# 8. Sauvegarde des statistiques
print("\n[7] Sauvegarde des statistiques...")
stats = {
    "model": "Baseline MNIST Classifier",
    "epochs": 5,
    "batch_size": 128,
    "architecture": "Dense(512)-Dropout(0.2)-Dense(10)",
    "results": {
        "training": {
            "loss": round(final_train_loss, 4),
            "accuracy": round(final_train_acc * 100, 2)
        },
        "validation": {
            "loss": round(final_val_loss, 4),
            "accuracy": round(final_val_acc * 100, 2)
        },
        "test": {
            "loss": round(test_loss, 4),
            "accuracy": round(test_accuracy * 100, 2)
        },
        "overfitting_gap": round(train_val_gap, 2),
        "diagnostic": diagnostic
    }
}

with open('ex1_statistics.json', 'w') as f:
    json.dump(stats, f, indent=4)
print("   - Statistiques sauvegardees: ex1_statistics.json")

# 9. Visualisation des courbes d'apprentissage
print("\n[8] Generation des graphiques...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Courbe de Loss
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o')
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
axes[0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoque', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Courbe d'Accuracy
axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
axes[1].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoque', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ex1_bias_variance_analysis.png', dpi=300, bbox_inches='tight')
print("   - Graphique sauvegarde : ex1_bias_variance_analysis.png")

plt.close()

# 10. Tableau récapitulatif
fig2, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Metric', 'Training', 'Validation', 'Test'],
    ['Loss', f"{final_train_loss:.4f}", f"{final_val_loss:.4f}", f"{test_loss:.4f}"],
    ['Accuracy', f"{final_train_acc*100:.2f}%", f"{final_val_acc*100:.2f}%", f"{test_accuracy*100:.2f}%"],
    ['', '', '', ''],
    ['Diagnostic', diagnostic, '', ''],
    ['Overfitting Gap', f"{train_val_gap:.2f}%", '', '']
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.23, 0.23, 0.23])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, 6):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('Exercise 1 - Summary Results', fontsize=16, fontweight='bold', pad=20)
plt.savefig('ex1_summary_table.png', dpi=300, bbox_inches='tight')
print("   - Tableau recapitulatif sauvegarde : ex1_summary_table.png")

plt.close()

# 11. Sauvegarde du modèle
model.save('ex1_baseline_model.h5')
print("\n   - Modele sauvegarde : ex1_baseline_model.h5")

print("\n" + "=" * 60)
print("EXERCICE 1 TERMINE")
print("=" * 60)
print("\n[FICHIERS GENERES]")
print("   1. ex1_bias_variance_analysis.png")
print("   2. ex1_summary_table.png")
print("   3. ex1_statistics.json")
print("   4. ex1_baseline_model.h5")
print("\n[QUESTION] : Votre modele souffre-t-il de bias ou variance ?")
print(f"   -> REPONSE : {diagnostic}")
print(f"   -> Ecart train-val : {train_val_gap:.2f}%")