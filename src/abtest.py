"""
Загрузка и подготовка данных A/B-теста new_payment_funnel_test_summer_2025.

Используется в ноутбуке notebooks/ab_test_analysis.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

TEST_NAME = "new_payment_funnel_test_summer_2025"
REGISTRATION_START = pd.Timestamp("2025-06-01")
REGISTRATION_END = pd.Timestamp("2025-06-14")   # последний день набора участников
TEST_END = pd.Timestamp("2025-06-22")           # события учитываем до конца 21 июня


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Читает четыре таблицы и сразу переводит даты в datetime."""
    files = {
        "users": ("new_users_test_2025.csv", ["first_date"]),
        "sessions": ("sessions_test_2025.csv", ["session_start"]),
        "purchases": ("purchases_test_2025.csv", ["event_dt"]),
        "participants": ("participants_test_2025.csv", []),
    }
    tables = {}
    for name, (file, dates) in files.items():
        path = raw_dir / file
        if not path.exists():
            raise FileNotFoundError(f"Не найден {file}. Положите файлы в data/raw/ (см. data/raw/README.md).")
        tables[name] = pd.read_csv(path, parse_dates=dates)
    return tables


def test_users(users: pd.DataFrame, participants: pd.DataFrame, test_name: str = TEST_NAME) -> pd.DataFrame:
    """Участники нужного теста с атрибутами регистрации (дата, регион, устройство)."""
    part = participants.loc[participants["ab_test"] == test_name, ["user_id", "group"]]
    return users.merge(part, on="user_id", how="inner")


def within_test(events: pd.DataFrame, date_col: str, members: pd.DataFrame) -> pd.DataFrame:
    """События участников теста, совершённые после регистрации и до окончания теста."""
    out = events.merge(members[["user_id", "group", "first_date"]], on="user_id", how="inner")
    return out[(out[date_col] >= out["first_date"]) & (out[date_col] < TEST_END)].copy()


def srm_test(counts: pd.Series) -> float:
    """Sample Ratio Mismatch: p-value хи-квадрат теста на равные размеры групп (план 50/50)."""
    return st.chisquare(counts.values).pvalue


def proportions_ztest(successes: pd.Series, totals: pd.Series) -> tuple[float, float]:
    """Двусторонний z-тест для двух долей. Возвращает (z, p-value)."""
    p1, p2 = successes.iloc[0] / totals.iloc[0], successes.iloc[1] / totals.iloc[1]
    pooled = successes.sum() / totals.sum()
    se = np.sqrt(pooled * (1 - pooled) * (1 / totals.iloc[0] + 1 / totals.iloc[1]))
    z = (p2 - p1) / se
    return z, 2 * st.norm.sf(abs(z))


def diff_ci(successes: pd.Series, totals: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал разности долей (вторая группа минус первая)."""
    p = successes / totals
    se = np.sqrt((p * (1 - p) / totals).sum())
    z = st.norm.ppf(0.5 + level / 2)
    d = p.iloc[1] - p.iloc[0]
    return d - z * se, d + z * se


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    """95%-й бутстреп-интервал разности средних (b − a) — для скошенной выручки."""
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n_boot)]
    return tuple(np.percentile(diffs, [2.5, 97.5]))
