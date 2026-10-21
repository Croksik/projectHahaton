"""Оркестрация: отчёты → оценки LLM → динамика."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import pandas as pd

from analytics import compute_dynamics
from data_loader import load_reports
from llm_scorer import Criterion, CriterionScore, GigaChatScorer, HeuristicScorer, load_criteria

ProgressCallback = Callable[[int, int, str], None]


def _scores_to_rows(
    case_id: str,
    volunteer_id: str,
    meeting_date: pd.Timestamp,
    scores: Sequence[CriterionScore],
) -> list[dict[str, object]]:
    return [
        {
            "id_кейса": case_id,
            "id_волонтера": volunteer_id,
            "дата_встречи": meeting_date,
            "criterion_id": item.criterion_id,
            "value": item.value,
            "evidence": item.evidence,
        }
        for item in scores
    ]


def build_scorer(
    criteria: Sequence[Criterion],
    use_llm: bool = True,
) -> GigaChatScorer | HeuristicScorer:
    """Создаёт скорер: GigaChat при наличии ключа, иначе эвристику."""
    if use_llm:
        return GigaChatScorer(criteria)
    return HeuristicScorer(criteria)


def analyze_reports(
    source,
    criteria_path: str | Path = "criteria.yaml",
    use_llm: bool = True,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Полный цикл анализа выгрузки.

    Returns:
        Кортеж ``(reports, scores, dynamics)``.
    """
    reports = load_reports(source)
    criteria = load_criteria(criteria_path)
    scorer = build_scorer(criteria, use_llm=use_llm)

    rows: list[dict[str, object]] = []
    total = len(reports)
    for i, row in reports.iterrows():
        case_id = str(row["id_кейса"])
        if progress:
            progress(int(i) + 1, total, case_id)
        try:
            scores = scorer.score_meeting(str(row["текст_отчета"]))
        except Exception as exc:
            scores = [
                CriterionScore(
                    criterion_id=c.id,
                    value=0,
                    evidence=f"Ошибка оценки: {exc}",
                )
                for c in criteria
            ]
        rows.extend(
            _scores_to_rows(
                case_id=case_id,
                volunteer_id=str(row["id_волонтера"]),
                meeting_date=row["дата_встречи"],
                scores=scores,
            )
        )

    scores_df = pd.DataFrame(rows)
    dynamics_df = compute_dynamics(scores_df) if not scores_df.empty else pd.DataFrame()
    return reports, scores_df, dynamics_df
