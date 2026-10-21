"""Загрузка и подготовка выгрузки отчётов наставников из Excel."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import BinaryIO

import pandas as pd

REQUIRED_COLUMNS = ("id_кейса", "id_волонтера", "дата_встречи", "текст_отчета")

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id_кейса": ("id_кейса", "case_id", "id_case", "кейс", "id кейса"),
    "id_волонтера": (
        "id_волонтера",
        "volunteer_id",
        "id_volunteer",
        "волонтер",
        "id волонтера",
        "наставник",
    ),
    "дата_встречи": ("дата_встречи", "meeting_date", "date", "дата", "дата встречи"),
    "текст_отчета": (
        "текст_отчета",
        "report_text",
        "text",
        "отчет",
        "текст отчета",
        "комментарий",
    ),
}

_WHITESPACE_RE = re.compile(r"\s+")


class _HTMLStripper(HTMLParser):
    """Собирает текстовые узлы, игнорируя разметку."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def strip_html(text: str | float | None) -> str:
    """Удаляет HTML-теги и схлопывает пробелы.

    Args:
        text: Сырой текст отчёта, возможно с HTML.

    Returns:
        Очищенная строка. Для пустых/NaN значений — пустая строка.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    raw = str(text)
    stripper = _HTMLStripper()
    try:
        stripper.feed(raw)
        stripper.close()
        cleaned = stripper.get_text()
    except Exception:
        cleaned = re.sub(r"<[^>]+>", " ", raw)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _resolve_columns(columns: list[str]) -> dict[str, str]:
    """Сопоставляет фактические имена колонок каноническим."""
    normalized = {_normalize_column_name(c): c for c in columns}
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            key = _normalize_column_name(alias)
            if key in normalized:
                found = normalized[key]
                break
        if found is None:
            missing.append(canonical)
        else:
            mapping[canonical] = found
    if missing:
        raise ValueError(
            "В Excel нет обязательных колонок: "
            + ", ".join(missing)
            + f". Фактические колонки: {list(columns)}"
        )
    return mapping


def load_reports(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Загружает Excel с отчётами, чистит HTML и сортирует встречи.

    Args:
        source: Путь к .xlsx или файловый объект (например, UploadFile Streamlit).

    Returns:
        DataFrame с каноническими колонками, отсортированный по
        ``id_кейса`` и ``дата_встречи``.
    """
    df = pd.read_excel(source, engine="openpyxl")
    mapping = _resolve_columns(list(df.columns))
    out = pd.DataFrame(
        {
            canonical: df[original]
            for canonical, original in mapping.items()
        }
    )
    out["id_кейса"] = out["id_кейса"].astype(str).str.strip()
    out["id_волонтера"] = out["id_волонтера"].astype(str).str.strip()
    out["дата_встречи"] = pd.to_datetime(out["дата_встречи"], errors="coerce")
    out["текст_отчета"] = out["текст_отчета"].map(strip_html)

    bad_dates = out["дата_встречи"].isna().sum()
    if bad_dates:
        out = out.dropna(subset=["дата_встречи"]).copy()

    out = out.sort_values(["id_кейса", "дата_встречи"], kind="mergesort")
    out = out.reset_index(drop=True)
    out["номер_встречи"] = out.groupby("id_кейса").cumcount() + 1
    return out
