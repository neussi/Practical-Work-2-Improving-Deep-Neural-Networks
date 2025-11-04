"""
Exercice 4: Batch Normalization
TP2 Deep Learning Engineering - ENSPY 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import json

# Configuration pour reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("EXERCICE 4 : BATCH NORMALIZATION")
print("=" * 70)

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

print(f"    Training: {x_train.shape[0]} samples")
print(f"    Validation: {x_val.shape[0]} samples")
print(f"    Test: {x_test.shape[0]} samples")

# 2. MODELE SANS BATCH NORMALIZATION (baseline)
print("\n" + "=" * 70)
print("[2] MODELE BASELINE (Sans Batch Normalization)")
print("=" * 70)

model_baseline = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
], name='baseline_no_bn')

model_baseline.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n    Architecture:")
model_baseline.summary()

print("\n[INFO] Entrainement du modele baseline...")
start_time = time.time()

history_baseline = model_baseline.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

baseline_time = time.time() - start_time
print(f"\n[INFO] Temps d'entrainement: {baseline_time:.2f}s")

# Evaluation
test_loss_baseline, test_acc_baseline = model_baseline.evaluate(x_test, y_test, verbose=0)
print(f"[INFO] Test Accuracy: {test_acc_baseline * 100:.2f}%")

# 3. MODELE AVEC BATCH NORMALIZATION
print("\n" + "=" * 70)
print("[3] MODELE AVEC BATCH NORMALIZATION")
print("=" * 70)

model_bn = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
], name='with_bn')

model_bn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n    Architecture avec BatchNormalization:")
model_bn.summary()

print("\n[INFO] Entrainement du modele avec Batch Normalization...")
start_time = time.time()

history_bn = model_bn.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

bn_time = time.time() - start_time
print(f"\n[INFO] Temps d'entrainement: {bn_time:.2f}s")

# Evaluation
test_loss_bn, test_acc_bn = model_bn.evaluate(x_test, y_test, verbose=0)
print(f"[INFO] Test Accuracy: {test_acc_bn * 100:.2f}%")

# 4. MODELE AVEC BATCH NORMALIZATION APRES CHAQUE COUCHE
print("\n" + "=" * 70)
print("[4] MODELE AVEC BATCH NORMALIZATION MULTIPLE")
print("=" * 70)

model_bn_multi = keras.Sequential([
    layers.Dense(512, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(10, activation='softmax')
], name='with_bn_multi')

model_bn_multi.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n    Architecture avec BatchNormalization multiple:")
model_bn_multi.summary()

print("\n[INFO] Entrainement du modele avec BN multiple...")
start_time = time.time()

history_bn_multi = model_bn_multi.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

bn_multi_time = time.time() - start_time
print(f"\n[INFO] Temps d'entrainement: {bn_multi_time:.2f}s")

# Evaluation
test_loss_bn_multi, test_acc_bn_multi = model_bn_multi.evaluate(x_test, y_test, verbose=0)
print(f"[INFO] Test Accuracy: {test_acc_bn_multi * 100:.2f}%")

# 5. MODELE AVEC BN ET LEARNING RATE PLUS ELEVE
print("\n" + "=" * 70)
print("[5] MODELE AVEC BN ET LEARNING RATE AUGMENTE")
print("=" * 70)

model_bn_highLR = keras.Sequential([
    layers.Dense(512, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(10, activation='softmax')
], name='with_bn_highLR')

# Learning rate augmenté grace a la stabilité de BN
model_bn_highLR.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n    Architecture avec BatchNormalization + LR=0.01:")
print("    Note: Learning rate 10x plus eleve (0.01 vs 0.001)")

print("\n[INFO] Entrainement avec learning rate augmente...")
start_time = time.time()

history_bn_highLR = model_bn_highLR.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

bn_highLR_time = time.time() - start_time
print(f"\n[INFO] Temps d'entrainement: {bn_highLR_time:.2f}s")

# Evaluation
test_loss_bn_highLR, test_acc_bn_highLR = model_bn_highLR.evaluate(x_test, y_test, verbose=0)
print(f"[INFO] Test Accuracy: {test_acc_bn_highLR * 100:.2f}%")

# 6. PREPARATION DES RESULTATS
results = [
    {
        'name': 'Baseline (No BN)',
        'test_acc': test_acc_baseline * 100,
        'train_time': baseline_time,
        'final_val_acc': history_baseline.history['val_accuracy'][-1] * 100,
        'final_train_acc': history_baseline.history['accuracy'][-1] * 100,
        'history': history_baseline
    },
    {
        'name': 'With BN (1 layer)',
        'test_acc': test_acc_bn * 100,
        'train_time': bn_time,
        'final_val_acc': history_bn.history['val_accuracy'][-1] * 100,
        'final_train_acc': history_bn.history['accuracy'][-1] * 100,
        'history': history_bn
    },
    {
        'name': 'With BN (Multi)',
        'test_acc': test_acc_bn_multi * 100,
        'train_time': bn_multi_time,
        'final_val_acc': history_bn_multi.history['val_accuracy'][-1] * 100,
        'final_train_acc': history_bn_multi.history['accuracy'][-1] * 100,
        'history': history_bn_multi
    },
    {
        'name': 'BN + High LR',
        'test_acc': test_acc_bn_highLR * 100,
        'train_time': bn_highLR_time,
        'final_val_acc': history_bn_highLR.history['val_accuracy'][-1] * 100,
        'final_train_acc': history_bn_highLR.history['accuracy'][-1] * 100,
        'history': history_bn_highLR
    }
]

# 7. SAUVEGARDE DES STATISTIQUES EN JSON
print("\n[6] Sauvegarde des statistiques...")
stats = {
    'models': []
}

for r in results:
    model_stats = {
        'name': r['name'],
        'test_accuracy': round(r['test_acc'], 2),
        'validation_accuracy': round(r['final_val_acc'], 2),
        'training_accuracy': round(r['final_train_acc'], 2),
        'training_time': round(r['train_time'], 2),
        'overfitting_gap': round(r['final_train_acc'] - r['final_val_acc'], 2)
    }
    stats['models'].append(model_stats)

with open('ex4_statistics.json', 'w') as f:
    json.dump(stats, f, indent=4)
print("    Statistiques sauvegardees: ex4_statistics.json")

# 8. TABLEAU COMPARATIF
print("\n" + "=" * 70)
print("TABLEAU COMPARATIF")
print("=" * 70)

print(f"\n{'Model':<25} {'Test Acc':<12} {'Val Acc':<12} {'Time':<12}")
print("-" * 70)
for r in results:
    print(f"{r['name']:<25} {r['test_acc']:>10.2f}%  {r['final_val_acc']:>10.2f}%  {r['train_time']:>9.2f}s")

# 9. ANALYSE DE LA VITESSE DE CONVERGENCE
print("\n[7] Analyse de la vitesse de convergence...")

convergence_epochs = []
for r in results:
    val_accs = r['history'].history['val_accuracy']
    epoch_95 = None
    for i, acc in enumerate(val_accs):
        if acc >= 0.95:
            epoch_95 = i + 1
            break
    convergence_epochs.append(epoch_95)
    status = str(epoch_95) if epoch_95 is not None else 'N/A'
    print(f"    {r['name']:<25} Atteint 95% val acc a l'epoch: {status}")

# 10. VISUALISATIONS COMPARATIVES
print("\n[8] Generation des graphiques...")

fig = plt.figure(figsize=(18, 12))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

# Graphique 1: Validation Accuracy Evolution
ax1 = plt.subplot(2, 3, 1)
for i, r in enumerate(results):
    ax1.plot(r['history'].history['val_accuracy'], 
             label=r['name'], linewidth=2.5, color=colors[i])
ax1.set_title('Validation Accuracy Evolution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Validation Accuracy', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Graphique 2: Training Loss Evolution
ax2 = plt.subplot(2, 3, 2)
for i, r in enumerate(results):
    ax2.plot(r['history'].history['loss'], 
             label=r['name'], linewidth=2.5, color=colors[i])
ax2.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Training Loss', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Graphique 3: Validation Loss Evolution
ax3 = plt.subplot(2, 3, 3)
for i, r in enumerate(results):
    ax3.plot(r['history'].history['val_loss'], 
             label=r['name'], linewidth=2.5, color=colors[i])
ax3.set_title('Validation Loss Evolution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Validation Loss', fontsize=11)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Graphique 4: Test Accuracy Comparison
ax4 = plt.subplot(2, 3, 4)
names = [r['name'] for r in results]
test_accs = [r['test_acc'] for r in results]
bars = ax4.bar(names, test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax4.set_ylabel('Test Accuracy (%)', fontsize=11)
ax4.set_ylim([min(test_accs) - 1, max(test_accs) + 1])
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
for i, v in enumerate(test_accs):
    ax4.text(i, v + 0.15, f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Graphique 5: Training Time Comparison
ax5 = plt.subplot(2, 3, 5)
times = [r['train_time'] for r in results]
bars = ax5.barh(names, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax5.set_xlabel('Time (seconds)', fontsize=11)
for i, v in enumerate(times):
    ax5.text(v + 1, i, f'{v:.1f}s', va='center', fontsize=10, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# Graphique 6: Convergence Speed
ax6 = plt.subplot(2, 3, 6)
conv_values = [e if e is not None else 10 for e in convergence_epochs]
bars = ax6.bar(names, conv_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_title('Convergence Speed (Epochs to 95%)', fontsize=14, fontweight='bold')
ax6.set_ylabel('Number of Epochs', fontsize=11)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=30, ha='right')
for i, (v, epoch) in enumerate(zip(conv_values, convergence_epochs)):
    text = str(epoch) if epoch is not None else 'N/A'
    ax6.text(i, v + 0.2, text, ha='center', fontsize=10, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ex4_batch_normalization_comparison.png', dpi=300, bbox_inches='tight')
print("    Graphique sauvegarde: ex4_batch_normalization_comparison.png")

plt.close()

# 11. GRAPHIQUE DETAILLE DE CONVERGENCE
fig2, ax = plt.subplots(figsize=(12, 6))
for i, r in enumerate(results):
    epochs_range = range(1, len(r['history'].history['val_accuracy']) + 1)
    ax.plot(epochs_range, r['history'].history['val_accuracy'], 
            marker='o', linewidth=2.5, markersize=6, color=colors[i], label=r['name'])

ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Target', alpha=0.7)
ax.set_title('Detailed Convergence Analysis', fontsize=16, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Validation Accuracy', fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.90, 1.0])

plt.tight_layout()
plt.savefig('ex4_convergence_detail.png', dpi=300, bbox_inches='tight')
print("    Graphique convergence sauvegarde: ex4_convergence_detail.png")

plt.close()

# 12. TABLEAU RECAPITULATIF EN IMAGE
fig3, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

table_data = [['Model', 'Test Acc', 'Val Acc', 'Train Acc', 'Time', 'Gap']]
for r in results:
    gap = r['final_train_acc'] - r['final_val_acc']
    table_data.append([
        r['name'],
        f"{r['test_acc']:.2f}%",
        f"{r['final_val_acc']:.2f}%",
        f"{r['final_train_acc']:.2f}%",
        f"{r['train_time']:.2f}s",
        f"{gap:.2f}%"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
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

plt.title('Summary Table - Batch Normalization Impact', fontsize=16, fontweight='bold', pad=20)
plt.savefig('ex4_summary_table.png', dpi=300, bbox_inches='tight')
print("    Tableau recapitulatif sauvegarde: ex4_summary_table.png")

plt.close()

# 13. ANALYSE DETAILLEE
print("\n" + "=" * 70)
print("ANALYSE DES RESULTATS")
print("=" * 70)

improvement_acc = test_acc_bn - test_acc_baseline
improvement_time = baseline_time - bn_time

print(f"\n[IMPACT DE BATCH NORMALIZATION]")
print(f"    Amelioration Test Accuracy: {improvement_acc * 100:+.2f}%")
print(f"    Difference temps entrainement: {improvement_time:+.2f}s")

print(f"\n[MODELE LE PLUS PERFORMANT]")
best = max(results, key=lambda x: x['test_acc'])
print(f"    Modele: {best['name']}")
print(f"    Test Accuracy: {best['test_acc']:.2f}%")

print(f"\n[CONVERGENCE LA PLUS RAPIDE]")
valid_convergences = [e for e in convergence_epochs if e is not None]
if valid_convergences:
    fastest_conv = min(valid_convergences)
    fastest_idx = convergence_epochs.index(fastest_conv)
    fastest_model = results[fastest_idx]['name']
    print(f"    Modele: {fastest_model}")
    print(f"    Epochs pour atteindre 95%: {fastest_conv}")
else:
    print(f"    Aucun modele n'a atteint 95% validation accuracy")

print("\n[AVANTAGES DE BATCH NORMALIZATION]")
print("    1. Stabilisation de l'entrainement")
print("    2. Permet des learning rates plus eleves")
print("    3. Acceleration de la convergence")
print("    4. Regularisation implicite (reduit overfitting)")
print("    5. Reduction de la sensibilite a l'initialisation")

print("\n[COMMENT CA MARCHE]")
print("    Pour chaque mini-batch:")
print("    1. Normalisation: z_norm = (z - mu) / sqrt(var + epsilon)")
print("    2. Scale et shift: z_out = gamma * z_norm + beta")
print("    3. gamma et beta sont appris pendant l'entrainement")

print("\n" + "=" * 70)
print("EXERCICE 4 TERMINE")
print("=" * 70)
print("\n[FICHIERS GENERES]")
print("    1. ex4_batch_normalization_comparison.png")
print("    2. ex4_convergence_detail.png")
print("    3. ex4_summary_table.png")
print("    4. ex4_statistics.json")
print("\n[CONCLUSION]")
print("    Batch Normalization est un outil puissant pour:")
print("    - Accelerer l'entrainement")
print("    - Ameliorer la stabilite")
print("    - Permettre des architectures plus profondes")
print("    - Obtenir de meilleures performances")