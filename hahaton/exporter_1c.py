"""Выгрузка результатов анализа в CSV для загрузки в 1С."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_SEP = ";"
CSV_ENCODING = "utf-8-sig"

MEETINGS_COLUMNS = ("id_кейса", "дата_встречи", "criterion_id", "value", "evidence")
DYNAMICS_COLUMNS = (
    "id_кейса",
    "id_волонтера",
    "criterion_id",
    "mean_score",
    "slope",
    "trend_status",
)


def _prepare_meetings(scores: pd.DataFrame) -> pd.DataFrame:
    missing = set(MEETINGS_COLUMNS) - set(scores.columns)
    if missing:
        raise ValueError(f"Для meetings_scores.csv не хватает колонок: {sorted(missing)}")
    out = scores.loc[:, list(MEETINGS_COLUMNS)].copy()
    out["дата_встречи"] = pd.to_datetime(out["дата_встречи"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0).astype(int)
    out["evidence"] = out["evidence"].fillna("").astype(str)
    return out


def _prepare_dynamics(dynamics: pd.DataFrame) -> pd.DataFrame:
    missing = set(DYNAMICS_COLUMNS) - set(dynamics.columns)
    if missing:
        raise ValueError(f"Для dynamics_summary.csv не хватает колонок: {sorted(missing)}")
    out = dynamics.loc[:, list(DYNAMICS_COLUMNS)].copy()
    out["mean_score"] = pd.to_numeric(out["mean_score"], errors="coerce").round(4)
    out["slope"] = pd.to_numeric(out["slope"], errors="coerce").round(6)
    return out


def export_to_1c(
    scores: pd.DataFrame,
    dynamics: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Сохраняет два CSV для CRM на базе 1С.

    Args:
        scores: Оценки по встречам (long-формат).
        dynamics: Сводка динамики по кейсам и критериям.
        output_dir: Каталог выгрузки.

    Returns:
        Пути к ``meetings_scores.csv`` и ``dynamics_summary.csv``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meetings_path = output_dir / "meetings_scores.csv"
    dynamics_path = output_dir / "dynamics_summary.csv"

    _prepare_meetings(scores).to_csv(
        meetings_path,
        sep=CSV_SEP,
        index=False,
        encoding=CSV_ENCODING,
    )
    _prepare_dynamics(dynamics).to_csv(
        dynamics_path,
        sep=CSV_SEP,
        index=False,
        encoding=CSV_ENCODING,
    )
    return meetings_path, dynamics_path
