"""Moduł ewaluacji modelu (lab. 3).

Uruchomienie: python evaluate.py

Skrypt wczytuje zapisany wcześniej model z katalogu models/ (nie trenuje go
ponownie) oraz dane testowe z data_prep.py, liczy metryki (accuracy,
classification_report) i zapisuje wynik do pliku tekstowego.
"""

from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from data_prep import get_data_config, get_env, load_data, split_data

load_dotenv()


def load_model(path):
    """Wczytuje model zapisany przez train.py."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Nie znaleziono modelu '{path}'. Najpierw uruchom: python train.py"
        )
    return joblib.load(path)


def evaluate_model(model, X_test, y_test):
    """Zwraca krotkę (accuracy, classification_report) dla danych testowych."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)


def save_results(path, accuracy, report):
    """Zapisuje metryki do pliku tekstowego (tworzy katalog, jeśli trzeba)."""
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "WYNIKI EWALUACJI MODELU (zbiór testowy)\n"
        "=======================================\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )
    path.write_text(text, encoding="utf-8")
    return text


def main():
    cfg = get_data_config()
    model_path = get_env("MODEL_PATH")
    results_path = get_env("RESULTS_PATH")

    model = load_model(model_path)
    df = load_data(cfg["data_path"])
    # te same parametry i ziarno co w train.py => identyczny zbiór testowy
    _, X_test, _, y_test = split_data(df, cfg["test_size"], cfg["seed"], cfg["target_column"])

    accuracy, report = evaluate_model(model, X_test, y_test)
    text = save_results(results_path, accuracy, report)
    print(text)
    print(f"Wyniki zapisano w: {results_path}")


if __name__ == "__main__":
    main()
