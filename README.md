# Rock-Paper-Scissors CV Recognizer

Kő-papír-olló gesztusfelismerő rendszer számítógépes látással.

## Telepítés

```bash
pip install -r requirements.txt
```

## Indítás

```bash
python src/app.py
```

Böngészőben megnyílik: `http://localhost:7860`

## Használat

1. Tölts fel képet vagy használd a webkamerát
2. Válassz beállításokat a legördülő menükből
3. Kattints a "Process Image" gombra

## Projekt struktúra

```
src/
├── core/                  # Pipeline és base osztályok
├── preprocessing/         # Képfeldolgozás (pl. GrayscaleConverter)
├── feature_extraction/    # Jellemzők kinyerése (pl. DummyGeometricExtractor)
├── classification/        # Osztályozók (pl. DummyClassifier)
├── ui/                    # Felhasználói felület
│   ├── components/        # UI komponensek (class-alapú)
│   ├── styles/           # CSS
│   ├── templates/        # Jinja2 HTML sablonok
│   └── utils/            # Template engine
└── app.py                # Indító fájl
```

## Állapot

- ✅ Architektúra kész
- ✅ Moduláris komponens rendszer
- ✅ Dummy modulok teszteléshez
- ⏳ ML modellek fejlesztés alatt