"""Динамика оценок по кейсам: среднее, наклон тренда, статус."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TrendStatus = Literal["рост", "снижение", "стабильно"]

SLOPE_GROWTH = 0.05
SLOPE_DECLINE = -0.05


def classify_trend(slope: float) -> TrendStatus:
    """Классифицирует наклон линейного тренда."""
    if slope > SLOPE_GROWTH:
        return "рост"
    if slope < SLOPE_DECLINE:
        return "снижение"
    return "стабильно"


def _slope(values: np.ndarray) -> float:
    """Линейный тренд по порядковому номеру встречи.

    При одной точке или константном ряде возвращает 0.0.
    """
    y = np.asarray(values, dtype=float)
    if y.size < 2 or np.allclose(y, y[0]):
        return 0.0
    x = np.arange(y.size, dtype=float)
    try:
        from scipy.stats import linregress

        result = linregress(x, y)
        return float(result.slope)
    except Exception:
        return float(np.polyfit(x, y, 1)[0])


def compute_dynamics(scores: pd.DataFrame) -> pd.DataFrame:
    """Считает динамику 0/1-оценок по каждому кейсу и критерию.

    Args:
        scores: Long-формат с колонками ``id_кейса``, ``id_волонтера``,
            ``дата_встречи``, ``criterion_id``, ``value``.

    Returns:
        Сводка: ``id_кейса``, ``id_волонтера``, ``criterion_id``,
        ``meetings_count``, ``mean_score``, ``slope``, ``trend_status``.
    """
    required = {"id_кейса", "id_волонтера", "дата_встречи", "criterion_id", "value"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"В таблице оценок нет колонок: {sorted(missing)}")

    work = scores.copy()
    work["дата_встречи"] = pd.to_datetime(work["дата_встречи"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce").fillna(0).astype(int)
    work = work.sort_values(["id_кейса", "criterion_id", "дата_встречи"], kind="mergesort")

    volunteer_map = (
        work.sort_values("дата_встречи")
        .groupby("id_кейса")["id_волонтера"]
        .last()
    )
    meetings_count = work.groupby("id_кейса")["дата_встречи"].nunique()

    rows: list[dict[str, object]] = []
    grouped = work.groupby(["id_кейса", "criterion_id"], sort=False)
    for (case_id, criterion_id), group in grouped:
        values = group["value"].to_numpy()
        slope = _slope(values)
        rows.append(
            {
                "id_кейса": case_id,
                "id_волонтера": volunteer_map.get(case_id, ""),
                "criterion_id": criterion_id,
                "meetings_count": int(meetings_count.get(case_id, len(values))),
                "mean_score": float(np.mean(values)),
                "slope": slope,
                "trend_status": classify_trend(slope),
            }
        )

    return pd.DataFrame(rows)
