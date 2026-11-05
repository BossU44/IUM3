"""Jednorazowy skrypt: zapisuje zbiór Wine (ten sam co w lab. 2) do pliku CSV.

Uruchomienie: python export_dataset.py
Ścieżka wyjściowa i nazwa kolumny z etykietą pochodzą z pliku .env.
"""

from pathlib import Path

from sklearn.datasets import load_wine

from data_prep import get_env


def main():
    data_path = Path(get_env("DATA_PATH"))
    target_column = get_env("TARGET_COLUMN")

    df = load_wine(as_frame=True).frame.rename(columns={"target": target_column})
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Zapisano {len(df)} wierszy do {data_path}")


if __name__ == "__main__":
    main()
