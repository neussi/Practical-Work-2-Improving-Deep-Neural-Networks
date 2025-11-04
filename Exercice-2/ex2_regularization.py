"""
Exercice 2: Application de la Regularisation (L2 + Dropout)
TP2 Deep Learning Engineering - ENSPY 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import json

# Configuration pour reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("EXERCICE 2 : REGULARISATION L2 + DROPOUT")
print("=" * 60)

# 1. Chargement et préparation des données
print("\n[1] Chargement et preparation des donnees...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Division train/val
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]

# Normalisation
x_train = x_train.reshape(54000, 784).astype("float32") / 255.0
x_val = x_val.reshape(6000, 784).astype("float32") / 255.0
x_test = x_test.reshape(10000, 784).astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"   - Training: {x_train.shape[0]} | Validation: {x_val.shape[0]} | Test: {x_test.shape[0]}")

# 2. MODÈLE SANS RÉGULARISATION (baseline)
print("\n" + "=" * 60)
print("[2] MODELE BASELINE (Sans regularisation)")
print("=" * 60)

model_baseline = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
], name='baseline')

model_baseline.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n   Entrainement du modele baseline...")
history_baseline = model_baseline.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

test_loss_baseline, test_acc_baseline = model_baseline.evaluate(x_test, y_test, verbose=0)

# 3. MODÈLE AVEC RÉGULARISATION L2
print("\n" + "=" * 60)
print("[3] MODELE AVEC L2 REGULARIZATION")
print("=" * 60)

model_l2 = keras.Sequential([
    layers.Dense(
        512, 
        activation='relu', 
        input_shape=(784,),
        kernel_regularizer=regularizers.l2(0.001)
    ),
    layers.Dropout(0.2),
    layers.Dense(
        10, 
        activation='softmax',
        kernel_regularizer=regularizers.l2(0.001)
    )
], name='l2_regularized')

model_l2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n   Architecture avec L2 (lambda=0.001):")
print("   - Dense(512) avec kernel_regularizer=l2(0.001)")
print("   - Dropout(0.2)")
print("   - Dense(10) avec kernel_regularizer=l2(0.001)")

print("\n   Entrainement du modele avec L2...")
history_l2 = model_l2.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

test_loss_l2, test_acc_l2 = model_l2.evaluate(x_test, y_test, verbose=0)

# 4. MODÈLE AVEC DROPOUT AUGMENTÉ
print("\n" + "=" * 60)
print("[4] MODELE AVEC DROPOUT AUGMENTE")
print("=" * 60)

model_dropout = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
], name='dropout_heavy')

model_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n   Architecture avec Dropout renforce:")
print("   - Dense(512) + Dropout(0.3)")
print("   - Dense(256) + Dropout(0.3)")
print("   - Dense(10)")

print("\n   Entrainement du modele avec Dropout augmente...")
history_dropout = model_dropout.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

test_loss_dropout, test_acc_dropout = model_dropout.evaluate(x_test, y_test, verbose=0)

# 5. MODÈLE AVEC L2 + DROPOUT COMBINÉS
print("\n" + "=" * 60)
print("[5] MODELE AVEC L2 + DROPOUT COMBINES")
print("=" * 60)

model_combined = keras.Sequential([
    layers.Dense(
        512, 
        activation='relu', 
        input_shape=(784,),
        kernel_regularizer=regularizers.l2(0.001)
    ),
    layers.Dropout(0.3),
    layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    ),
    layers.Dropout(0.3),
    layers.Dense(
        10, 
        activation='softmax',
        kernel_regularizer=regularizers.l2(0.001)
    )
], name='combined')

model_combined.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n   Architecture combinee (L2 + Dropout):")
print("   - Dense(512) + L2 + Dropout(0.3)")
print("   - Dense(256) + L2 + Dropout(0.3)")
print("   - Dense(10) + L2")

print("\n   Entrainement du modele combine...")
history_combined = model_combined.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

test_loss_combined, test_acc_combined = model_combined.evaluate(x_test, y_test, verbose=0)

# 6. PREPARATION DES RESULTATS
models_results = {
    'Baseline (Dropout 0.2)': (model_baseline, history_baseline, test_loss_baseline, test_acc_baseline),
    'L2 Regularization': (model_l2, history_l2, test_loss_l2, test_acc_l2),
    'Heavy Dropout (0.3)': (model_dropout, history_dropout, test_loss_dropout, test_acc_dropout),
    'L2 + Dropout Combined': (model_combined, history_combined, test_loss_combined, test_acc_combined)
}

comparison_data = []

print("\n" + "=" * 60)
print("COMPARAISON DES PERFORMANCES")
print("=" * 60)

for name, (model, history, test_loss, test_acc) in models_results.items():
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = (final_train_acc - final_val_acc) * 100
    
    comparison_data.append({
        'name': name,
        'train_acc': final_train_acc * 100,
        'val_acc': final_val_acc * 100,
        'test_acc': test_acc * 100,
        'test_loss': test_loss,
        'gap': overfitting_gap,
        'history': history
    })
    
    print(f"\n{name}:")
    print(f"   Train Accuracy: {final_train_acc * 100:.2f}%")
    print(f"   Val Accuracy:   {final_val_acc * 100:.2f}%")
    print(f"   Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"   Overfitting Gap: {overfitting_gap:.2f}%")

# 7. SAUVEGARDE DES STATISTIQUES
print("\n[6] Sauvegarde des statistiques...")
stats = {
    "experiment": "Regularization Comparison",
    "models": []
}

for data in comparison_data:
    model_stats = {
        "name": data['name'],
        "train_accuracy": round(data['train_acc'], 2),
        "validation_accuracy": round(data['val_acc'], 2),
        "test_accuracy": round(data['test_acc'], 2),
        "test_loss": round(data['test_loss'], 4),
        "overfitting_gap": round(data['gap'], 2)
    }
    stats["models"].append(model_stats)

with open('ex2_statistics.json', 'w') as f:
    json.dump(stats, f, indent=4)
print("   - Statistiques sauvegardees: ex2_statistics.json")

# 8. VISUALISATION COMPARATIVE
print("\n[7] Generation des graphiques comparatifs...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Graphique 1: Validation Loss Comparison
ax = axes[0, 0]
for data in comparison_data:
    ax.plot(data['history'].history['val_loss'], label=data['name'], linewidth=2)
ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoque')
ax.set_ylabel('Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Graphique 2: Validation Accuracy Comparison
ax = axes[0, 1]
for data in comparison_data:
    ax.plot(data['history'].history['val_accuracy'], label=data['name'], linewidth=2)
ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoque')
ax.set_ylabel('Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Graphique 3: Overfitting Gap
ax = axes[1, 0]
names = [d['name'] for d in comparison_data]
gaps = [d['gap'] for d in comparison_data]
colors = ['red' if g > 3 else 'green' for g in gaps]
ax.barh(names, gaps, color=colors, alpha=0.7)
ax.set_xlabel('Overfitting Gap (%)')
ax.set_title('Overfitting Analysis (Train - Val Accuracy)', fontsize=14, fontweight='bold')
ax.axvline(x=3, color='orange', linestyle='--', linewidth=2, label='Threshold (3%)')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Graphique 4: Test Accuracy Comparison
ax = axes[1, 1]
test_accs = [d['test_acc'] for d in comparison_data]
ax.bar(names, test_accs, color='steelblue', alpha=0.7)
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([min(test_accs) - 2, max(test_accs) + 1])
for i, v in enumerate(test_accs):
    ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

plt.tight_layout()
plt.savefig('ex2_regularization_comparison.png', dpi=300, bbox_inches='tight')
print("   - Graphique sauvegarde : ex2_regularization_comparison.png")

plt.close()

# 9. TABLEAU RECAPITULATIF
fig2, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

table_data = [['Model', 'Test Acc', 'Val Acc', 'Train Acc', 'Gap']]
for d in comparison_data:
    table_data.append([
        d['name'],
        f"{d['test_acc']:.2f}%",
        f"{d['val_acc']:.2f}%",
        f"{d['train_acc']:.2f}%",
        f"{d['gap']:.2f}%"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('Regularization Impact Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig('ex2_summary_table.png', dpi=300, bbox_inches='tight')
print("   - Tableau recapitulatif sauvegarde: ex2_summary_table.png")

plt.close()

# 10. CONCLUSION
print("\n" + "=" * 60)
print("ANALYSE ET CONCLUSION")
print("=" * 60)

best_model = min(comparison_data, key=lambda x: x['gap'])
print(f"\n[MEILLEUR MODELE] (plus faible overfitting): {best_model['name']}")
print(f"   Overfitting Gap: {best_model['gap']:.2f}%")
print(f"   Test Accuracy: {best_model['test_acc']:.2f}%")

print("\n[IMPACT DE LA REGULARISATION]")
print("   - L2 regularization: Penalise les poids eleves -> poids plus petits")
print("   - Dropout: Desactive aleatoirement des neurones -> redondance")
print("   - Combinaison: Meilleure generalisation et reduction overfitting")

print("\n" + "=" * 60)
print("EXERCICE 2 TERMINE")
print("=" * 60)
print("\n[FICHIERS GENERES]")
print("   1. ex2_regularization_comparison.png")
print("   2. ex2_summary_table.png")
print("   3. ex2_statistics.json")
print("\n[QUESTION] : La regularisation ameliore-t-elle les performances ?")
print(f"   -> Comparez l'ecart train-val entre baseline et modeles regularises")
print(f"   -> Un gap plus faible indique une meilleure generalisation")