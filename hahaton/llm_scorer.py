"""Оценка отчетов наставников по критериям с помощью GigaChat.

Модуль читает criteria.yaml, формирует промпт, вызывает GigaChat,
валидирует ответ через Pydantic и повторяет запрос при невалидном JSON.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

import yaml
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------

class Criterion(BaseModel):
    """Один критерий из criteria.yaml."""
    id: str
    name: str
    description: str
    rubric_1: str
    rubric_0: str


class CriterionScore(BaseModel):
    """Оценка одного критерия по одному отчету."""
    criterion_id: str = Field(..., description="ID критерия из criteria.yaml")
    value: int = Field(..., description="0 или 1")
    evidence: str = Field(default="", description="Цитата из отчета")

    @field_validator("value")
    @classmethod
    def _binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"value должен быть 0 или 1, получено {v!r}")
        return v

    @field_validator("evidence")
    @classmethod
    def _clean_evidence(cls, v: str) -> str:
        return (v or "").strip()[:500]


class MeetingScores(RootModel[list[CriterionScore]]):
    """Результат оценки одной встречи — список оценок по всем критериям."""


class ScoringError(RuntimeError):
    """Не удалось получить валидные оценки после всех попыток."""


# ---------------------------------------------------------------------------
# Загрузка критериев и промпты
# ---------------------------------------------------------------------------

def load_criteria(path: str | Path = "criteria.yaml") -> list[Criterion]:
    """Загружает критерии из YAML и проверяет уникальность id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл критериев не найден: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = data.get("criteria") if isinstance(data, dict) else None
    if not raw:
        raise ValueError("В criteria.yaml нет ключа 'criteria' или список пуст")
    criteria = [Criterion(**item) for item in raw]
    ids = [c.id for c in criteria]
    if len(ids) != len(set(ids)):
        raise ValueError("В criteria.yaml есть дублирующиеся id")
    return criteria


_SYSTEM_PROMPT = """Ты — эксперт-методист фонда, который анализирует отчеты наставников.
Твоя задача — по тексту отчета о встрече оценить каждый критерий бинарно: 1 или 0.

Правила:
- 1 — если в тексте есть явное действие/обсуждение, соответствующее rubric_1.
- 0 — если тема не упоминается или упоминается без действий наставника (rubric_0).
- В поле evidence приводи короткую точную цитату из отчета (до 300 символов),
  которая обосновывает оценку. Для 0 — пустая строка или краткое пояснение.
- Не придумывай факты, опирайся только на текст отчета.
- Ответ — строго JSON-массив без markdown-разметки:
  [{"criterion_id": "...", "value": 0|1, "evidence": "..."}, ...]
- В массиве должны быть ВСЕ перечисленные критерии, ровно по одному разу.
"""


def _format_criteria_block(criteria: Sequence[Criterion]) -> str:
    lines = []
    for c in criteria:
        lines.append(
            f'- id: "{c.id}"\n'
            f"  name: {c.name}\n"
            f"  description: {c.description}\n"
            f"  rubric_1: {c.rubric_1}\n"
            f"  rubric_0: {c.rubric_0}"
        )
    return "\n".join(lines)


def build_prompt(report_text: str, criteria: Sequence[Criterion]) -> str:
    """Собирает пользовательский промпт для одной встречи."""
    return (
        "Критерии оценки:\n"
        f"{_format_criteria_block(criteria)}\n\n"
        "Текст отчета о встрече:\n"
        '"""\n'
        f"{report_text.strip()}\n"
        '"""\n\n'
        "Верни JSON-массив оценок по всем критериям."
    )


# ---------------------------------------------------------------------------
# Извлечение JSON из ответа модели
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_payload(text: str) -> list[dict[str, Any]]:
    """Достает JSON-массив из ответа модели (в т.ч. из markdown-блока)."""
    candidates: list[str] = []
    if m := _FENCE_RE.search(text):
        candidates.append(m.group(1))
    candidates.append(text)
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_err: Exception | None = None
    for raw in candidates:
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(data, dict) and "scores" in data:
            data = data["scores"]
        if isinstance(data, list):
            return data
        last_err = ValueError("JSON не является массивом оценок")
    raise ValueError(f"Не удалось извлечь JSON из ответа: {last_err}")


# ---------------------------------------------------------------------------
# Скоратор
# ---------------------------------------------------------------------------

class GigaChatScorer:
    """Обертка над GigaChat для оценки отчетов по критериям."""

    def __init__(
        self,
        criteria: Sequence[Criterion],
        model: str | None = None,
        max_retries: int = 3,
        temperature: float = 0.0,
        backoff_base: float = 2.0,
    ) -> None:
        self.criteria = list(criteria)
        self.expected_ids = {c.id for c in self.criteria}
        self.model = model or os.getenv("GIGACHAT_MODEL", "GigaChat")
        self.max_retries = max_retries
        self.temperature = temperature
        self.backoff_base = backoff_base
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from gigachat import GigaChat
        except ImportError as e:
            raise RuntimeError(
                "Не установлен пакет gigachat. Выполните: pip install gigachat"
            ) from e

        credentials = os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("GIGACHAT_TOKEN")
        if not credentials:
            raise RuntimeError(
                "Не задан GIGACHAT_CREDENTIALS (или GIGACHAT_TOKEN) в окружении/.env"
            )

        verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL", "0") == "1"
        ca_bundle = os.getenv("GIGACHAT_CA_BUNDLE")  # путь к сертификату Минцифры

        kwargs: dict[str, Any] = {
            "credentials": credentials,
            "scope": os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            "model": self.model,
            "verify_ssl_certs": verify_ssl,
            "timeout": float(os.getenv("GIGACHAT_TIMEOUT", "120")),
        }
        if ca_bundle:
            kwargs["ca_bundle_file"] = ca_bundle

        self._client = GigaChat(**kwargs)
        return self._client

    def _call_model(self, user_prompt: str) -> str:
        client = self._get_client()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        response = client.chat(payload)
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            raise ScoringError(f"Неожиданный формат ответа GigaChat: {response!r}") from e

    def _validate_scores(self, raw_scores: list[dict[str, Any]]) -> list[CriterionScore]:
        scores = [CriterionScore(**item) for item in raw_scores]
        got_ids = [s.criterion_id for s in scores]
        if len(got_ids) != len(set(got_ids)):
            raise ValueError("В ответе есть дублирующиеся criterion_id")
        missing = self.expected_ids - set(got_ids)
        extra = set(got_ids) - self.expected_ids
        if missing or extra:
            raise ValueError(
                f"Несовпадение критериев. missing={sorted(missing)}, extra={sorted(extra)}"
            )
        by_id = {s.criterion_id: s for s in scores}
        return [by_id[c.id] for c in self.criteria]

    def score_meeting(self, report_text: str) -> list[CriterionScore]:
        """Оценивает один отчет. Возвращает оценки по всем критериям в порядке YAML."""
        if not report_text or not report_text.strip():
            return [
                CriterionScore(criterion_id=c.id, value=0, evidence="Отчет пуст")
                for c in self.criteria
            ]

        base_prompt = build_prompt(report_text, self.criteria)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            prompt = base_prompt
            if last_error is not None:
                prompt += (
                    "\n\nВНИМАНИЕ: предыдущая попытка вернула невалидный ответ. "
                    f"Ошибка: {last_error}. Верни строго корректный JSON-массив "
                    "со всеми критериями."
                )
            try:
                raw = self._call_model(prompt)
                payload = _extract_json_payload(raw)
                return self._validate_scores(payload)
            except (ValidationError, ValueError, ScoringError) as e:
                last_error = e
                time.sleep(self.backoff_base ** (attempt - 1))
            except Exception as e:  # сетевые/API ошибки тоже ретраим
                last_error = e
                time.sleep(self.backoff_base ** (attempt - 1))

        raise ScoringError(
            f"Не удалось получить валидные оценки за {self.max_retries} попыток: {last_error}"
        )


class HeuristicScorer:
    """Демо-скорер без LLM: 1, если в тексте встречаются ключевые слова критерия."""

    def __init__(self, criteria: Sequence[Criterion]) -> None:
        self.criteria = list(criteria)

    def score_meeting(self, report_text: str) -> list[CriterionScore]:
        text = (report_text or "").lower()
        if not text.strip():
            return [
                CriterionScore(criterion_id=c.id, value=0, evidence="Отчет пуст")
                for c in self.criteria
            ]
        results: list[CriterionScore] = []
        for criterion in self.criteria:
            needles = _keywords_for(criterion)
            hit = next((word for word in needles if word and word in text), "")
            if hit:
                start = text.find(hit)
                snippet = report_text[max(0, start - 40) : start + 120].strip()
                results.append(
                    CriterionScore(
                        criterion_id=criterion.id,
                        value=1,
                        evidence=snippet,
                    )
                )
            else:
                results.append(
                    CriterionScore(
                        criterion_id=criterion.id,
                        value=0,
                        evidence="",
                    )
                )
        return results


def _keywords_for(criterion: Criterion) -> list[str]:
    """Грубые ключевые слова для демо-режима (не замена LLM)."""
    extra = {
        "leisure_hobbies": ["хобби", "кружок", "парк", "кино", "книг", "увлечен"],
        "emotional_support": ["поддерж", "эмоц", "тревог", "настроен", "успоко"],
        "family_relations": ["мама", "папа", "родств", "бабуш", "опекун"],
        "social_relations": ["друг", "сверст", "общен", "коллектив"],
        "family_values": ["семь", "ценност"],
        "household_skills": ["готов", "уборк", "быт", "покуп", "стирк"],
        "career_guidance": ["професс", "колледж", "ваканс", "специальн"],
        "employment": ["работ", "собеседован", "резюме", "смен"],
        "education": ["школ", "урок", "оценк", "домашн", "учёб", "учеб"],
        "health": ["здоров", "врач", "спорт", "тренир", "больн"],
        "addictions": ["куре", "алкогол", "сигарет", "зависимо"],
        "romantic_relations": ["девушк", "отношен", "симпат", "парень"],
        "financial_literacy": ["деньг", "бюджет", "копил", "зарплат", "руб"],
        "legal_literacy": ["закон", "документ", "паспорт", "права"],
        "safety_risks": ["мошен", "опасн", "риск", "безопас"],
        "future_plans": ["план", "будущ", "цель", "выпуск"],
        "wellbeing_growth": ["уверен", "инициатив", "спокойн"],
        "socialization_growth": ["подруж", "компани", "обща"],
        "independence_growth": ["сам", "самостоят"],
    }
    words = extra.get(criterion.id, [])
    words.append(criterion.name.lower())
    return words


def gigachat_configured() -> bool:
    """True, если в окружении заданы учётные данные GigaChat."""
    return bool(os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("GIGACHAT_TOKEN"))


def score_report(
    report_text: str,
    criteria_path: str | Path = "criteria.yaml",
    use_llm: bool | None = None,
) -> list[CriterionScore]:
    """Оценивает один отчёт по YAML-критериям.

    Args:
        report_text: Очищенный текст встречи.
        criteria_path: Путь к criteria.yaml.
        use_llm: Принудительно включить/выключить GigaChat.
            ``None`` — использовать LLM, если заданы credentials.
    """
    criteria = load_criteria(criteria_path)
    if use_llm is None:
        use_llm = gigachat_configured()
    scorer: GigaChatScorer | HeuristicScorer
    scorer = GigaChatScorer(criteria) if use_llm else HeuristicScorer(criteria)
    return scorer.score_meeting(report_text)


if __name__ == "__main__":
    sample = (
        "Встретились с Сашей в парке. Обсуждали школу: он жаловался, что "
        "трудно с математикой, я предложил позаниматься в следующий раз. "
        "Потом говорили про его увлечение робототехникой, он хочет пойти "
        "на кружок. Настроение у Саши было хорошее, он был активен."
    )
    criteria = load_criteria("criteria.yaml")
    scorer = GigaChatScorer(criteria)
    result = scorer.score_meeting(sample)
    print(json.dumps([s.model_dump() for s in result], ensure_ascii=False, indent=2))