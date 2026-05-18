"""Student-owned dataset loading contract.

Students must implement ``load_dataset_split`` so that ``scripts/main.py`` can
evaluate every configured model on the same test split.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR

PROCESSED_CSV = DATA_DIR / "processed" / "allego_labeled.csv"

FEATURE_COLS = [
    "puissance_nominale",
    "prise_type_ef",
    "prise_type_2",
    "prise_type_combo_ccs",
    "prise_type_chademo",
    "prise_type_autre",
    "implantation_encoded",
    "acces_libre",
    "nbre_pdc",
    "latitude",
    "longitude",
]


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.

    Typical choices for the return types are ``pandas.DataFrame`` /
    ``pandas.Series`` or ``numpy.ndarray``.
    """

    df = pd.read_csv(PROCESSED_CSV)
    X = df[FEATURE_COLS]
    y = df["label"]
    return tuple(train_test_split(X, y, test_size=0.2, random_state=42, stratify=y))
