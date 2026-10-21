"""Офлайн-проверки пайплайна без GigaChat."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics import classify_trend, compute_dynamics
from data_loader import load_reports, strip_html
from exporter_1c import export_to_1c
from llm_scorer import HeuristicScorer, load_criteria

ROOT = Path(__file__).resolve().parents[1]


def test_strip_html() -> None:
    raw = "<p>Привет,&nbsp;<b>Саша</b></p>"
    assert "Привет" in strip_html(raw)
    assert "<" not in strip_html(raw)


def test_load_and_score_sample(tmp_path: Path) -> None:
    sample = ROOT / "sample_reports.xlsx"
    reports = load_reports(sample)
    assert list(reports.columns)[:4] == [
        "id_кейса",
        "id_волонтера",
        "дата_встречи",
        "текст_отчета",
    ]
    for _, group in reports.groupby("id_кейса"):
        assert group["дата_встречи"].is_monotonic_increasing

    criteria = load_criteria(ROOT / "criteria.yaml")
    assert len(criteria) == 19
    scorer = HeuristicScorer(criteria)
    scores = scorer.score_meeting(reports.iloc[0]["текст_отчета"])
    assert {s.criterion_id for s in scores} == {c.id for c in criteria}
    assert all(s.value in (0, 1) for s in scores)


def test_dynamics_and_export(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        [
            {
                "id_кейса": "C1",
                "id_волонтера": "V1",
                "дата_встречи": "2026-01-01",
                "criterion_id": "emotional_support",
                "value": 0,
                "evidence": "",
            },
            {
                "id_кейса": "C1",
                "id_волонтера": "V1",
                "дата_встречи": "2026-02-01",
                "criterion_id": "emotional_support",
                "value": 1,
                "evidence": "поддержал",
            },
            {
                "id_кейса": "C1",
                "id_волонтера": "V1",
                "дата_встречи": "2026-03-01",
                "criterion_id": "emotional_support",
                "value": 1,
                "evidence": "успокоил",
            },
        ]
    )
    dyn = compute_dynamics(scores)
    assert dyn.iloc[0]["trend_status"] == classify_trend(float(dyn.iloc[0]["slope"]))
    assert dyn.iloc[0]["meetings_count"] == 3
    m_path, d_path = export_to_1c(scores, dyn, tmp_path)
    assert m_path.exists() and d_path.exists()
    loaded = pd.read_csv(m_path, sep=";")
    assert list(loaded.columns)[:3] == ["id_кейса", "дата_встречи", "criterion_id"]
