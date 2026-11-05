# Lab. 3 – Strukturyzacja projektu ML pod kątem integracji (CI)

Pipeline z lab. 2 (zbiór Wine, `StandardScaler` + `LogisticRegression`) podzielony na niezależne moduły
konfigurowane przez plik `.env`.

## Struktura

| Plik | Rola |
|------|------|
| `data_prep.py` | `load_data(path)`, `split_data(df, test_size, seed)` – bez efektów ubocznych |
| `train.py` | trening Pipeline i zapis modelu do `models/model.joblib` |
| `evaluate.py` | wczytanie modelu, metryki, zapis do `wyniki.txt` |
| `export_dataset.py` | jednorazowy eksport zbioru Wine do `data/wine.csv` |
| `tests/test_data_prep.py` | testy jednostkowe (pytest) |
| `.env.example` | wzór konfiguracji (klucze bez wartości) |

## Uruchomienie (z katalogu głównego projektu)

```bash
pip install -r requirements.txt
cp .env.example .env        # uzupełnij wartości (przykłady w komentarzach)
python export_dataset.py    # tworzy data/wine.csv (plik jest też w repozytorium)
python train.py             # trenuje i zapisuje model
python evaluate.py          # liczy metryki i zapisuje wyniki.txt
pytest                      # testy jednostkowe
```

Parametry można nadpisać zmiennymi środowiskowymi bez zmiany kodu, np. `TEST_SIZE=0.3 python train.py`.
