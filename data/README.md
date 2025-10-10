# Data Directory

Dataset fájlok és feldolgozott adatok.

## 📁 Mappastruktúra

```
data/
├── raw/              # Eredeti dataset képek
│   ├── rock/         # Kő gesztus képek
│   ├── paper/        # Papír gesztus képek
│   └── scissors/     # Olló gesztus képek
│
├── splits/           # Train/val/test split indexek
│   └── split_indices.json
│
└── processed/        # Feldolgozott feature-ök, cache
    └── (későbbi használatra)
```

## 🚀 Setup

```bash
# 1. Dataset letöltés
python scripts/download_dataset.py --source kaggle-drgfreeman

# 2. Split létrehozás
python scripts/create_splits.py
```

## 📄 split_indices.json Formátum

```json
{
  "metadata": {
    "total_samples": 2188,
    "test_size": 0.2,
    "val_size": 0.1
  },
  "image_paths": ["rock/rock_0001.jpg", ...],
  "labels": [0, 1, 2, ...],
  "splits": {
    "train": [0, 5, 8, ...],
    "val": [1, 3, ...],
    "test": [2, 4, ...]
  }
}
```

**Labels:** `0=rock, 1=paper, 2=scissors`

## ⚠️ .gitignore

- `raw/**` - Dataset képek nem kerülnek verziókezelésbe
- `splits/**` - Split fájlok nem kerülnek verziókezelésbe
- `processed/**` - Feldolgozott adatok nem kerülnek verziókezelésbe

Csak `.gitkeep` fájlok tracked-eltek a mappastruktúra megőrzéséhez.