# Scripts

Python szkriptek dataset kezeléshez, model training-hez és teszteléshez.

## Struktúra

```
scripts/
├── data/              # Dataset kezelés
│   ├── download_dataset.py
│   ├── dataset_info.py
│   └── create_splits.py
├── training/          # Model training
│   └── mediapipe/
│       └── train_rf.py
└── tests/             # Tesztek
    └── mediapipe/
        ├── test_model.py
        └── ...
```

---

# Dataset Management (data/)

## 📥 Dataset Letöltés

```bash
# Elérhető források listázása
venv/bin/python scripts/data/download_dataset.py --list

# Letöltés Kaggle-ről
venv/bin/python scripts/data/download_dataset.py --source kaggle-drgfreeman
venv/bin/python scripts/data/download_dataset.py --source kaggle-sanikamal

# Custom output mappa
venv/bin/python scripts/data/download_dataset.py --source kaggle-drgfreeman --output data/custom
```

**Elérhető források:**
- `kaggle-drgfreeman`: 2188 kép, változatos háttér
- `kaggle-sanikamal`: 2520 kép, egyszerű háttér

## 📊 Dataset Információk

```bash
# Statisztikák: képszám, formátumok, osztály eloszlás
venv/bin/python scripts/data/dataset_info.py

# Custom mappa
venv/bin/python scripts/data/dataset_info.py --data data/custom
```

## 🔀 Train/Val/Test Split

```bash
# Alapértelmezett: 70% train, 10% val, 20% test
venv/bin/python scripts/data/create_splits.py

# Custom arányok
venv/bin/python scripts/data/create_splits.py --test-size 0.15 --val-size 0.15

# Custom output
venv/bin/python scripts/data/create_splits.py --output data/splits/custom_split.json
```

**Output:** `data/splits/split_indices.json`

## 🔧 Új Dataset Forrás Hozzáadása

1. Készíts új downloader-t: `scripts/downloaders/my_downloader.py`

```python
from .base_downloader import BaseDownloader

class MyDownloader(BaseDownloader):
    def download(self, target_dir: str) -> bool:
        # Implementáció
        return True
    
    def get_description(self) -> str:
        return "Dataset leírás"
```

2. Regisztráld `download_dataset.py`-ban:

```python
DOWNLOADERS = {
    'my-source': lambda: MyDownloader(),
}
```

3. Használat:

```bash
python scripts/download_dataset.py --source my-source
```

## 📁 Mappastruktúra

```
scripts/
├── download_dataset.py      # Fő orchestrator
├── dataset_info.py           # Statisztikák
├── create_splits.py          # Split létrehozás
├── downloaders/
│   ├── base_downloader.py    # Base osztály
│   └── kaggle_downloader.py  # Kaggle implementáció
└── utils/
    └── dataset_organizer.py  # Közös szervező logika
```

## ⚙️ Követelmények

```bash
pip install kagglehub scikit-learn
```

**Kaggle API config:** `~/.kaggle/kaggle.json`

Letöltés: https://www.kaggle.com/settings → Create API Token

---

# Model Training (training/)

## MediaPipe + Random Forest

```bash
# Train model (outputs: models/rf_mediapipe.pkl + features)
venv/bin/python scripts/training/mediapipe/train_rf.py
```

**Output:**
- `models/rf_mediapipe.pkl` - Trained Random Forest model
- `data/mediapipe_features/*.npy` - Extracted features for reuse

---

# Testing (tests/)

## MediaPipe Tests

```bash
# Model accuracy test
venv/bin/python scripts/tests/mediapipe/test_model.py --num-samples 20

# Full pipeline integration test
venv/bin/python scripts/tests/mediapipe/test_integration.py

# Check failed detections
venv/bin/python scripts/tests/mediapipe/check_failed_images.py

# Debug specific image
venv/bin/python scripts/tests/mediapipe/debug_hand_detection.py data/raw/rock/rock_0001.png

# Analyze detection issues
venv/bin/python scripts/tests/mediapipe/visualize_detection_issue.py data/raw/rock/rock_0009.png
```

Részletek: `scripts/tests/mediapipe/README.md`