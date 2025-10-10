# Rock-Paper-Scissors CV Recognizer

Kő-papír-olló gesztusfelismerő rendszer számítógépes látással.

## Telepítés

```bash
pip install gradio numpy opencv-python jinja2
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
├── core/              # Pipeline és base osztályok
├── preprocessing/     # Képfeldolgozás
├── feature_extraction/# Jellemzők kinyerése
├── classification/    # Osztályozók
├── ui/               # Felhasználói felület
└── app.py            # Indító fájl
```

## Állapot

- ✅ Architektúra kész
- ⏳ ML modellek fejlesztés alatt
