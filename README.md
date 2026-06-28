# Land Cover Classification — EuroSAT

Classifying satellite images from the [EuroSAT dataset](https://github.com/phelber/EuroSAT) using Random Forest and XGBoost. Since tree-based models can't consume raw images, RGB mean and standard deviation are extracted per channel and used as features.

## What's in the notebook

- Feature extraction from raw satellite images (per-channel mean + std via DataLoader)
- Train/test split with stratification + StandardScaler normalization
- Random Forest and XGBoost classifiers with tuned hyperparameters
- Accuracy, F1-score, and full classification report for both models
- Confusion matrices side by side
- Random sample prediction grid with true vs predicted labels
- Bar chart comparing Accuracy and F1 (macro) across both models

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 labeled satellite images across 10 land cover classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake). Images are 64×64 RGB patches from Sentinel-2.

Download and extract so the folder structure looks like:

```
2750/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

Then update the `root` path in the notebook:
```python
dataset = datasets.ImageFolder(root=r'path/to/2750', transform=transform)
```

## Setup

```bash
pip install numpy torch torchvision scikit-learn xgboost matplotlib seaborn tqdm
```

## Results

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Random Forest | 0.771 | 0.761 |
| XGBoost | 0.780 | 0.773 |

## Outputs

Running the notebook produces:
- `confusion_matrices.png`
- `predictions_RandomForestClassifier.png`
- `predictions_XGBClassifier.png`
- `model_comparison.png`

## Notes

The feature set (6 numbers per image — RGB mean + std) is intentionally minimal. Adding more per-channel statistics (median, min, max) or swapping to pretrained CNN embeddings would likely push accuracy up significantly.
