# Wytrenowane modele FBNet

Wagi rodziny FBNet z pracy magisterskiej *Zwiększanie płynności wideo poprzez
interpolację klatek z wykorzystaniem konwolucyjnych sieci neuronowych*
(AGH, 2026, Kamil Barszczak, prom. dr inż. Rafał Mrówka).

## Zawartość katalogów

| Katalog | Plik wag | Framework | Format | Historia uczenia |
|---|---|---|---|---|
| `model_v1`   | `FBNet.h5`  | TensorFlow / Keras | HDF5 | — |
| `model_v2`   | `FBNet.h5`  | TensorFlow / Keras | HDF5 | — |
| `model_v3`   | `FBNet.h5`  | TensorFlow / Keras | HDF5 | — |
| `model_v4`   | `FBNet.h5`  | TensorFlow / Keras | HDF5 | — |
| `model_v5`   | `fbnet.pt`  | PyTorch | state_dict | `history.pickle` |
| `model_v6`   | `fbnet.pt`  | PyTorch | state_dict | `history.pickle` |
| `model_v6_1` | `fbnet.pt`  | PyTorch | state_dict | `history.pickle` |
| `model_v6_7` | `fbnet.pt`  | PyTorch | state_dict | — |
| `model_v7_1` | `fbnet.pt`  | PyTorch | state_dict | — |

Kod definiujący architektury: `src/tensorflow/model_vN/` (v1–v4) oraz
`src/pytorch/model_vN/` (v5–v7.1).

## Co wnosi każda wersja

| Wersja | Parametry | Zmiana względem poprzedniej |
|---|---|---|
| v1   | 1 199 971 | model odniesienia: dwa osobne enkodery, dwie osobne sieci przepływu |
| v2   | 1 580 047 | enkoder syjamski (wspólne wagi), oszczędność przeznaczona na szersze warstwy |
| v3   | 1 965 047 | spójna konwencja znaku przepływu, większe pole recepcyjne estymatora (5×5 zamiast 3×3) |
| v4   | 985 977   | spiętrzanie cech w stylu FILM (enkoder trójkolumnowy), jeden symetryczny estymator wywoływany dwukrotnie |
| v5   | 1 916 383 | szerszy enkoder, PReLU, piramida obrazu z filtrem antyaliasingowym |
| v6   | 2 839 189 | bramki uwagi na skrótach dekodera, przemieszczanie jednej klatki, sploty z krokiem 2, dylatacja |
| v6.1 | 1 401 410 | powrót do poprawek rezydualnych, dropout, przemieszczanie połowicznym przepływem |
| v6.7 | 4 079 554 | flow attention — bramką steruje samo pole przepływu |
| v7.1 | 5 421 771 | rozdzielenie estymacji przepływu od kodowania cech, jednokrotne przemieszczanie w pełnej rozdzielczości |

Warianty v6.2–v6.6 powstawały eksploracyjnie, poza jednolitym protokołem —
opisane w pracy jakościowo, wagi nie są archiwizowane.

## Wyniki (Vimeo90K triplet, 3 782 próbki testowe)

| Model | Parametry | PSNR [dB] | SSIM | LPIPS ↓ | FloLPIPS ↓ | Czas [ms] |
|---|---|---|---|---|---|---|
| **v3** | 1 965 047 | **33,093** | **0,9450** | 0,0759 | 0,0740 | 27,6 |
| v5   | 1 916 383 | 32,380 | 0,9390 | **0,0625** | 0,0618 | 29,5 |
| v6.7 | 4 079 554 | 32,103 | 0,9283 | 0,0708 | **0,0610** | 40,8 |
| v4   | 985 977   | 32,030 | 0,9252 | 0,0977 | 0,0877 | 46,2 |
| v2   | 1 580 047 | 31,034 | 0,9078 | 0,1223 | 0,0963 | 45,4 |
| v7.1 | 5 421 771 | 30,753 | 0,8995 | 0,0968 | 0,0889 | 66,6 |
| *mieszanie klatek* | 0 | 25,607 | 0,7531 | 0,1702 | — | — |

v1, v6 i v6.1 nie przeszły pełnego protokołu ewaluacji — wagi zachowane
wyłącznie jako zapis przebiegu prac.

Ranking zależy od metryki: PSNR/SSIM → v3, LPIPS → v5, FloLPIPS → v6.7.

## Warunki uczenia (wspólne dla wszystkich ocenianych wersji)

- Zbiór: Vimeo90K triplet, 448×256, predykcja klatki środkowej (t = 0,5)
- Budżet: 30 epok × 1000 kroków × batch 8 = 240 000 próbek (≈ 4,7 epoki pełnego zbioru)
- Optymalizator: NAdam
- Sprzęt: NVIDIA RTX A6000 24 GB, 3,6–10,2 h na model
- Strata: `L = 5·L₁ + 10·L₂ + L_PSNR + w_s·L_str`, gdzie `L_PSNR = 1 − PSNR/40`
  - człon strukturalny `L_str`: `1 − SSIM` dla v1–v4 (w = 1,0),
    strata percepcyjna VGG16 po 4 blokach dla v5–v7.1 (w = 0,5 lub 1,0)

## Wczytanie wag

```python
# v1-v4 (TensorFlow)
import tensorflow as tf
model = tf.keras.models.load_model("models/model_v3/FBNet.h5", compile=False)

# v5-v7.1 (PyTorch)
import torch
from src.pytorch.model_v5.model import FBNet
model = FBNet()
model.load_state_dict(torch.load("models/model_v5/fbnet.pt", map_location="cpu"))
model.eval()
```
