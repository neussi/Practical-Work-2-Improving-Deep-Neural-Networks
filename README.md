
---

# Practical-Work-2 – Improving Deep Neural Networks

## Contenu du projet

Ce dépôt contient les implémentations des quatre exercices du TP2 visant à analyser l’amélioration des performances des réseaux de neurones profonds à travers différents concepts : biais-variance, régularisation, optimiseurs, et batch normalization.

---

## Récapitulatif des exercices

### Exercice 1 – `ex1_bias_variance.py`

Ce script génère les fichiers suivants :

* `ex1_bias_variance_analysis.png`
* `ex1_summary_table.png`
* `ex1_statistics.json`

---

### Exercice 2 – `ex2_regularization.py`

Ce script génère les fichiers suivants :

* `ex2_regularization_comparison.png`
* `ex2_summary_table.png`
* `ex2_statistics.json`

---

### Exercice 3 – `ex3_optimizers_mlflow.py`

Ce script réalise une comparaison de six optimiseurs et journalise les résultats via MLflow.

Sorties générées :

* `ex3_optimizers_comparison.png`
* Tracking MLflow (consultable via `mlflow ui`)

Instructions :

1. Créer le fichier `ex3_optimizers_mlflow.py`
2. Exécuter :

   ```bash
   python ex3_optimizers_mlflow.py
   ```
3. Lancer MLflow UI dans un terminal séparé :

   ```bash
   mlflow ui
   ```
4. Ouvrir :

   ```
   http://localhost:5000
   ```

Captures attendues :

* `capture_ex3_terminal.png` : sortie console avec les 6 optimiseurs
* `capture_ex3_graphs.png` : graphiques de comparaison
* `capture_ex3_mlflow_ui.png` : interface MLflow listant les runs
* `capture_ex3_mlflow_comparison.png` : page de comparaison des runs

Ce code teste 6 optimiseurs et journalise l’ensemble des résultats dans MLflow.

---

### Exercice 4 – `ex4_batch_normalization.py`

Ce script génère les fichiers suivants :

* `ex4_batch_normalization_comparison.png`
* `ex4_convergence_detail.png`
* `ex4_summary_table.png`
* `ex4_statistics.json`

---

## Exécution

```bash
python ex1_bias_variance.py
python ex2_regularization.py
python ex3_optimizers_mlflow.py
python ex4_batch_normalization.py
```
