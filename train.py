"""Moduł trenowania modelu (lab. 3).

Uruchomienie: python train.py

Skrypt wczytuje dane (data_prep.py), trenuje Pipeline (StandardScaler +
klasyfikator) i zapisuje wytrenowany model do katalogu models/ w formacie
.joblib. Wszystkie parametry pochodzą z pliku .env.
"""

import time
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import get_data_config, get_env, load_data, split_data

load_dotenv()


def build_model(model_type, seed, max_iter, c):
    """Buduje (jeszcze nietrenowany) Pipeline: skaler + klasyfikator."""
    if model_type != "logistic_regression":
        raise ValueError(
            f"Nieobsługiwany MODEL_TYPE: '{model_type}'. Dostępne: logistic_regression."
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=c, random_state=seed, max_iter=max_iter)),
        ]
    )


def train_model(X_train, y_train, model_type, seed, max_iter, c):
    """Trenuje model na danych treningowych i zwraca wytrenowany Pipeline."""
    model = build_model(model_type, seed, max_iter, c)
    model.fit(X_train, y_train)
    return model


def save_model(model, path):
    """Zapisuje model do pliku .joblib (tworzy katalog docelowy, jeśli trzeba)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def main():
    cfg = get_data_config()
    model_type = get_env("MODEL_TYPE")
    max_iter = get_env("MAX_ITER", int)
    c = get_env("C", float)
    model_path = get_env("MODEL_PATH")

    df = load_data(cfg["data_path"])
    X_train, X_test, y_train, y_test = split_data(
        df, cfg["test_size"], cfg["seed"], cfg["target_column"]
    )
    print(f"Dane: treningowe={len(X_train)}, testowe={len(X_test)}, cechy={X_train.shape[1]}")

    start = time.time()
    model = train_model(X_train, y_train, model_type, cfg["seed"], max_iter, c)
    print(f"Czas treningu ({model_type}): {time.time() - start:.4f} s")

    saved_to = save_model(model, model_path)
    print(f"Model zapisano w: {saved_to}")


if __name__ == "__main__":
    main()
