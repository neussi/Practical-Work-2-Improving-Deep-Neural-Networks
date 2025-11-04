"""
Exercice 3: Comparaison des Optimizers avec MLflow
TP2 Deep Learning Engineering - ENSPY 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.keras
import time
import json
from datetime import datetime

# Configuration pour reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("EXERCICE 3 : COMPARAISON DES OPTIMIZERS AVEC MLFLOW")
print("=" * 70)

# Configuration MLflow
mlflow.set_experiment("TP2_Optimizer_Comparison")
print("\n[INFO] MLflow experiment: TP2_Optimizer_Comparison")

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

# 2. Définition des optimizers à comparer
print("\n[2] Configuration des optimizers...")

optimizers_config = {
    'SGD_basic': keras.optimizers.SGD(learning_rate=0.01),
    'SGD_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'SGD_nesterov': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': keras.optimizers.Adam(learning_rate=0.001),
    'AdamW': keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
}

print(f"    Nombre d'optimizers a tester: {len(optimizers_config)}")
for name in optimizers_config.keys():
    print(f"    - {name}")

# 3. Fonction pour créer le modèle
def create_model():
    """Crée un modèle avec architecture standardisée"""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 4. Boucle d'entraînement avec MLflow tracking
print("\n" + "=" * 70)
print("DEBUT DES EXPERIMENTATIONS")
print("=" * 70)

results = []
EPOCHS = 10
BATCH_SIZE = 128

for opt_name, optimizer in optimizers_config.items():
    print(f"\n{'=' * 70}")
    print(f"[EXPERIMENT] Optimizer: {opt_name}")
    print(f"{'=' * 70}")
    
    # Démarrer un run MLflow
    with mlflow.start_run(run_name=f"Optimizer_{opt_name}"):
        
        # Log des paramètres
        mlflow.log_param("optimizer_name", opt_name)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("architecture", "Dense(512)-Dropout(0.2)-Dense(10)")
        mlflow.log_param("dataset", "MNIST")
        mlflow.log_param("train_samples", 54000)
        mlflow.log_param("val_samples", 6000)
        
        # Créer et compiler le modèle
        model = create_model()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n[INFO] Compilation avec {opt_name}")
        
        # Entraînement avec chronométrage
        print(f"[INFO] Debut entrainement ({EPOCHS} epoques)...")
        start_time = time.time()
        
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Évaluation sur le test set
        print(f"\n[INFO] Evaluation sur le test set...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # Calcul des métriques finales
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        # Log des métriques dans MLflow
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("overfitting_gap", final_train_acc - final_val_acc)
        
        # Log des courbes d'apprentissage (epoch par epoch)
        for epoch in range(EPOCHS):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Sauvegarder le modèle comme artifact
        mlflow.keras.log_model(model, "model")
        
        # Affichage des résultats
        print(f"\n[RESULTATS] {opt_name}:")
        print(f"    Train Accuracy: {final_train_acc * 100:.2f}%")
        print(f"    Val Accuracy:   {final_val_acc * 100:.2f}%")
        print(f"    Test Accuracy:  {test_acc * 100:.2f}%")
        print(f"    Training Time:  {training_time:.2f}s")
        print(f"    Overfitting Gap: {(final_train_acc - final_val_acc) * 100:.2f}%")
        
        # Stocker les résultats pour comparaison
        results.append({
            'optimizer': opt_name,
            'train_acc': final_train_acc * 100,
            'val_acc': final_val_acc * 100,
            'test_acc': test_acc * 100,
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'test_loss': test_loss,
            'time': training_time,
            'gap': (final_train_acc - final_val_acc) * 100,
            'history': history.history
        })
        
        print(f"[INFO] Run MLflow complete pour {opt_name}")

# 5. SAUVEGARDE DES STATISTIQUES
print("\n[5] Sauvegarde des statistiques...")
stats = {
    "experiment": "Optimizer Comparison",
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizers": []
}

for r in results:
    optimizer_stats = {
        "name": r['optimizer'],
        "test_accuracy": round(r['test_acc'], 2),
        "validation_accuracy": round(r['val_acc'], 2),
        "training_accuracy": round(r['train_acc'], 2),
        "test_loss": round(r['test_loss'], 4),
        "training_time": round(r['time'], 2),
        "overfitting_gap": round(r['gap'], 2)
    }
    stats["optimizers"].append(optimizer_stats)

with open('ex3_statistics.json', 'w') as f:
    json.dump(stats, f, indent=4)
print("    Statistiques sauvegardees: ex3_statistics.json")

# 6. Tableau comparatif
print("\n" + "=" * 70)
print("TABLEAU COMPARATIF DES OPTIMIZERS")
print("=" * 70)

print(f"\n{'Optimizer':<20} {'Test Acc':<12} {'Val Acc':<12} {'Time':<12} {'Gap':<10}")
print("-" * 70)
for r in results:
    print(f"{r['optimizer']:<20} {r['test_acc']:>10.2f}%  {r['val_acc']:>10.2f}%  {r['time']:>9.2f}s  {r['gap']:>8.2f}%")

# 7. Visualisations comparatives
print("\n[6] Generation des graphiques comparatifs...")

fig = plt.figure(figsize=(18, 12))

# Graphique 1: Validation Accuracy Curves
ax1 = plt.subplot(2, 3, 1)
for r in results:
    ax1.plot(r['history']['val_accuracy'], label=r['optimizer'], linewidth=2)
ax1.set_title('Validation Accuracy Evolution', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Accuracy')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Graphique 2: Training Loss Curves
ax2 = plt.subplot(2, 3, 2)
for r in results:
    ax2.plot(r['history']['loss'], label=r['optimizer'], linewidth=2)
ax2.set_title('Training Loss Evolution', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Training Loss')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Graphique 3: Test Accuracy Comparison
ax3 = plt.subplot(2, 3, 3)
optimizers_names = [r['optimizer'] for r in results]
test_accs = [r['test_acc'] for r in results]
colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
bars = ax3.bar(optimizers_names, test_accs, color=colors, alpha=0.8)
ax3.set_title('Final Test Accuracy', fontsize=13, fontweight='bold')
ax3.set_ylabel('Test Accuracy (%)')
ax3.set_ylim([min(test_accs) - 2, max(test_accs) + 1])
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
for i, v in enumerate(test_accs):
    ax3.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Graphique 4: Training Time Comparison
ax4 = plt.subplot(2, 3, 4)
times = [r['time'] for r in results]
bars = ax4.barh(optimizers_names, times, color=colors, alpha=0.8)
ax4.set_title('Training Time', fontsize=13, fontweight='bold')
ax4.set_xlabel('Time (seconds)')
for i, v in enumerate(times):
    ax4.text(v + 1, i, f'{v:.1f}s', va='center', fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

# Graphique 5: Overfitting Gap
ax5 = plt.subplot(2, 3, 5)
gaps = [r['gap'] for r in results]
colors_gap = ['red' if g > 3 else 'green' for g in gaps]
ax5.barh(optimizers_names, gaps, color=colors_gap, alpha=0.7)
ax5.set_title('Overfitting Gap (Train - Val)', fontsize=13, fontweight='bold')
ax5.set_xlabel('Gap (%)')
ax5.axvline(x=3, color='orange', linestyle='--', linewidth=2, label='Threshold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='x')

# Graphique 6: Convergence Speed
ax6 = plt.subplot(2, 3, 6)
convergence_epochs = []
for r in results:
    val_accs = r['history']['val_accuracy']
    epoch_95 = None
    for i, acc in enumerate(val_accs):
        if acc >= 0.95:
            epoch_95 = i
            break
    convergence_epochs.append(epoch_95 if epoch_95 is not None else EPOCHS)

bars = ax6.bar(optimizers_names, convergence_epochs, color=colors, alpha=0.8)
ax6.set_title('Convergence Speed (Epochs to 95% Val Acc)', fontsize=13, fontweight='bold')
ax6.set_ylabel('Epochs')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
for i, v in enumerate(convergence_epochs):
    label = f'{v}' if v < EPOCHS else 'N/A'
    ax6.text(i, v + 0.2, label, ha='center', fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ex3_optimizers_comparison.png', dpi=300, bbox_inches='tight')
print("    Graphique sauvegarde: ex3_optimizers_comparison.png")

plt.close()

# 8. TABLEAU RECAPITULATIF EN IMAGE
fig2, ax = plt.subplots(figsize=(14, 5))
ax.axis('tight')
ax.axis('off')

table_data = [['Optimizer', 'Test Acc', 'Val Acc', 'Train Acc', 'Time', 'Gap']]
for r in results:
    table_data.append([
        r['optimizer'],
        f"{r['test_acc']:.2f}%",
        f"{r['val_acc']:.2f}%",
        f"{r['train_acc']:.2f}%",
        f"{r['time']:.2f}s",
        f"{r['gap']:.2f}%"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.20, 0.13, 0.13, 0.13, 0.13, 0.13])
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

plt.title('Optimizer Comparison Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig('ex3_summary_table.png', dpi=300, bbox_inches='tight')
print("    Tableau recapitulatif sauvegarde: ex3_summary_table.png")

plt.close()

# 9. Analyse et recommandations
print("\n" + "=" * 70)
print("ANALYSE DES RESULTATS")
print("=" * 70)

best_accuracy = max(results, key=lambda x: x['test_acc'])
fastest = min(results, key=lambda x: x['time'])
best_generalization = min(results, key=lambda x: x['gap'])

print(f"\n[MEILLEURE ACCURACY] {best_accuracy['optimizer']}")
print(f"    Test Accuracy: {best_accuracy['test_acc']:.2f}%")

print(f"\n[PLUS RAPIDE] {fastest['optimizer']}")
print(f"    Training Time: {fastest['time']:.2f}s")

print(f"\n[MEILLEURE GENERALISATION] {best_generalization['optimizer']}")
print(f"    Overfitting Gap: {best_generalization['gap']:.2f}%")

print("\n[RECOMMANDATIONS]")
print("    SGD basic: Lent, necessite tuning du learning rate")
print("    SGD momentum: Ameliore SGD, accelere convergence")
print("    SGD nesterov: Version amelioree du momentum")
print("    RMSprop: Adapte le learning rate, bon pour RNN")
print("    Adam: Choix par defaut, combine Momentum + RMSprop")
print("    AdamW: Adam avec weight decay, meilleure regularisation")

print("\n" + "=" * 70)
print("EXERCICE 3 TERMINE")
print("=" * 70)
print("\n[FICHIERS GENERES]")
print("    1. ex3_optimizers_comparison.png")
print("    2. ex3_summary_table.png")
print("    3. ex3_statistics.json")
print("    4. MLflow tracking (mlruns/)")
print("\n[INFO] Consultez MLflow UI pour visualiser les resultats:")
print("    mlflow ui")
print("    Puis ouvrez: http://localhost:5000")