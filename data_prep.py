"""Moduł przygotowania danych (lab. 3).

Odpowiada wyłącznie za wczytanie danych i podział na zbiór treningowy
i testowy. Funkcje `load_data` oraz `split_data` nie mają efektów ubocznych
(nie drukują niczego i nie zapisują plików), dzięki czemu można je
importować i testować niezależnie od reszty projektu.

Parametry (ścieżki, rozmiar zbioru testowego, ziarno losowości) są
wczytywane z pliku .env przy pomocy python-dotenv.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Wczytanie zmiennych z pliku .env (jeśli istnieje). Zmienne środowiskowe
# ustawione już wcześniej (np. w CI) mają pierwszeństwo przed plikiem .env.
load_dotenv()


def get_env(key, cast=str):
    """Zwraca wartość zmiennej `key` z .env / środowiska, rzutowaną funkcją `cast`.

    Gdy zmienna nie jest ustawiona (lub jest pusta), rzuca RuntimeError
    z czytelnym komunikatem, zamiast po cichu używać wartości wpisanej w kodzie.
    """
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise RuntimeError(
            f"Brak zmiennej '{key}'. Skopiuj .env.example do .env i uzupełnij wartości."
        )
    return cast(value.strip())


def get_data_config():
    """Zwraca parametry związane z danymi, wczytane z .env."""
    return {
        "data_path": get_env("DATA_PATH"),
        "target_column": get_env("TARGET_COLUMN"),
        "test_size": get_env("TEST_SIZE", float),
        "seed": get_env("RANDOM_SEED", int),
    }


def load_data(path):
    """Wczytuje dane z pliku CSV i zwraca DataFrame.

    Raises:
        FileNotFoundError: gdy plik nie istnieje.
        ValueError: gdy plik jest pusty lub zawiera tylko nagłówek.
    """
    # pandas.errors.EmptyDataError (plik 0 B) jest podklasą ValueError.
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Plik '{path}' nie zawiera żadnych wierszy z danymi.")
    return df


def split_data(df, test_size, seed, target_column=None):
    """Dzieli DataFrame na zbiory treningowy i testowy (podział stratyfikowany).

    Args:
        df: DataFrame z cechami i kolumną z etykietą klasy.
        test_size: ułamek danych przeznaczony na zbiór testowy, z przedziału (0, 1).
        seed: ziarno losowości (powtarzalność podziału).
        target_column: nazwa kolumny z etykietą; domyślnie ostatnia kolumna.

    Returns:
        Krotka (X_train, X_test, y_train, y_test).

    Raises:
        TypeError: gdy `df` nie jest DataFrame.
        ValueError: gdy dane są puste, `test_size` jest poza (0, 1)
            lub brakuje kolumny z etykietą.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument 'df' musi być obiektem pandas.DataFrame.")
    if df.empty:
        raise ValueError("DataFrame jest pusty - nie ma czego dzielić.")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size musi należeć do przedziału (0, 1), podano: {test_size}.")

    if target_column is None:
        target_column = df.columns[-1]
    if target_column not in df.columns:
        raise ValueError(f"Brak kolumny z etykietą '{target_column}' w danych.")
    if df.shape[1] < 2:
        raise ValueError("Dane muszą zawierać co najmniej jedną cechę oraz kolumnę z etykietą.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # stratify=y zachowuje proporcje klas (tak samo jak w notatniku z lab. 2)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
