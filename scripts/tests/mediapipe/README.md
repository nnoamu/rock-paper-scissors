# Test Scripts

Test és debug szkriptek a MediaPipe + Random Forest modell teszteléséhez.

## Használat

Minden szkriptet a **projekt root**-ból kell futtatni:

```bash
# Model teszt
venv/bin/python scripts/tests/mediapipe/test_model.py --num-samples 20

# Full pipeline integration teszt
venv/bin/python scripts/tests/mediapipe/test_integration.py

# Sikertelen detektálások vizsgálata
venv/bin/python scripts/tests/mediapipe/check_failed_images.py

# Egyedi kép debug
venv/bin/python scripts/tests/mediapipe/debug_hand_detection.py data/raw/rock/rock_0001.png

# Részletes elemzés
venv/bin/python scripts/tests/mediapipe/visualize_detection_issue.py data/raw/rock/rock_0009.png
```

## Szkriptek

### `test_model.py`
Random test képeken teszteli a MediaPipe + RF modellt.
- **Paraméter**: `--num-samples N` (default: 5)
- **Output**: Predikciók, confidence, accuracy

### `test_integration.py`
Teljes pipeline integration teszt (preprocessing → extraction → classification).

### `check_failed_images.py`
Listázza azokat a képeket ahol a MediaPipe nem detektált kezet.

### `debug_hand_detection.py`
Részletes elemzés egy adott képről (miért nem detektál kezet).

### `visualize_detection_issue.py`
Több kép elemzése egyszerre, részletes info.

## Tipikus eredmények

- **Detection rate**: ~82% (MediaPipe talál kezet)
- **Classification accuracy**: 100% (amikor talál kezet)
- **Overall accuracy**: ~82%
