# Dataset Management Scripts

Dataset letöltés, szervezés és split létrehozás scriptek.

## 📥 Dataset Letöltés

```bash
# Elérhető források listázása
python scripts/download_dataset.py --list

# Letöltés Kaggle-ről
python scripts/download_dataset.py --source kaggle-drgfreeman
python scripts/download_dataset.py --source kaggle-sanikamal

# Custom output mappa
python scripts/download_dataset.py --source kaggle-drgfreeman --output data/custom
```

**Elérhető források:**
- `kaggle-drgfreeman`: 2188 kép, változatos háttér
- `kaggle-sanikamal`: 2520 kép, egyszerű háttér

## 📊 Dataset Információk

```bash
# Statisztikák: képszám, formátumok, osztály eloszlás
python scripts/dataset_info.py

# Custom mappa
python scripts/dataset_info.py --data data/custom
```

## 🔀 Train/Val/Test Split

```bash
# Alapértelmezett: 70% train, 10% val, 20% test
python scripts/create_splits.py

# Custom arányok
python scripts/create_splits.py --test-size 0.15 --val-size 0.15

# Custom output
python scripts/create_splits.py --output data/splits/custom_split.json
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