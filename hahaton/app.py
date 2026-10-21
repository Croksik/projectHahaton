"""Streamlit-интерфейс анализа отчётов наставников."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from exporter_1c import export_to_1c
from llm_scorer import gigachat_configured, load_criteria
from pipeline import analyze_reports

ROOT = Path(__file__).resolve().parent
CRITERIA_PATH = ROOT / "criteria.yaml"
EXPORT_DIR = ROOT / "exports"

st.set_page_config(
    page_title="Аналитика наставничества",
    page_icon="🤝",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _criteria_table() -> pd.DataFrame:
    items = load_criteria(CRITERIA_PATH)
    return pd.DataFrame(
        [{"id": c.id, "name": c.name, "description": c.description} for c in items]
    )


def _run_analysis(uploaded, use_llm: bool) -> None:
    progress_bar = st.sidebar.progress(0, text="Загрузка отчётов…")
    status = st.sidebar.empty()

    def on_progress(done: int, total: int, case_id: str) -> None:
        ratio = done / max(total, 1)
        progress_bar.progress(ratio, text=f"{done}/{total} встреч")
        status.caption(f"Кейс {case_id}")

    reports, scores, dynamics = analyze_reports(
        uploaded,
        criteria_path=CRITERIA_PATH,
        use_llm=use_llm,
        progress=on_progress,
    )
    progress_bar.progress(1.0, text="Готово")
    status.caption(f"Обработано встреч: {len(reports)}")

    meetings_path, dynamics_path = export_to_1c(scores, dynamics, EXPORT_DIR)
    st.session_state["reports"] = reports
    st.session_state["scores"] = scores
    st.session_state["dynamics"] = dynamics
    st.session_state["export_meetings"] = meetings_path.read_bytes()
    st.session_state["export_dynamics"] = dynamics_path.read_bytes()


def _macro_tab(scores: pd.DataFrame, dynamics: pd.DataFrame, labels: dict[str, str]) -> None:
    st.subheader("Общая сводка")
    n_cases = scores["id_кейса"].nunique()
    n_meetings = scores[["id_кейса", "дата_встречи"]].drop_duplicates().shape[0]
    mean_all = float(scores["value"].mean()) if not scores.empty else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Кейсов", n_cases)
    c2.metric("Встреч", n_meetings)
    c3.metric("Средний балл по всем критериям", f"{mean_all:.2f}")

    by_criterion = (
        scores.groupby("criterion_id", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_score"})
    )
    by_criterion["Критерий"] = by_criterion["criterion_id"].map(labels).fillna(
        by_criterion["criterion_id"]
    )
    fig_bar = px.bar(
        by_criterion.sort_values("mean_score"),
        x="mean_score",
        y="Критерий",
        orientation="h",
        title="Средние баллы по критериям (все кейсы)",
        range_x=[0, 1],
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    trend_counts = (
        dynamics["trend_status"].value_counts().rename_axis("Тренд").reset_index(name="Количество")
        if not dynamics.empty
        else pd.DataFrame(columns=["Тренд", "Количество"])
    )
    fig_trend = px.pie(
        trend_counts,
        names="Тренд",
        values="Количество",
        title="Распределение трендов",
        color="Тренд",
        color_discrete_map={
            "рост": "#2ca02c",
            "стабильно": "#7f7f7f",
            "снижение": "#d62728",
        },
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("#### Сводка по кейсам")
    summary = (
        dynamics.groupby(["id_кейса", "id_волонтера"], as_index=False)
        .agg(
            meetings_count=("meetings_count", "max"),
            mean_score=("mean_score", "mean"),
            growth=("trend_status", lambda s: int((s == "рост").sum())),
            decline=("trend_status", lambda s: int((s == "снижение").sum())),
        )
        .sort_values("mean_score", ascending=False)
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _micro_tab(scores: pd.DataFrame, dynamics: pd.DataFrame, labels: dict[str, str]) -> None:
    st.subheader("Карточка пары наставник — подопечный")
    cases = sorted(scores["id_кейса"].unique())
    case_id = st.selectbox("id_кейса", cases)
    criteria_ids = sorted(scores["criterion_id"].unique())
    criterion_id = st.selectbox(
        "Критерий",
        criteria_ids,
        format_func=lambda cid: labels.get(cid, cid),
    )

    case_dyn = dynamics[
        (dynamics["id_кейса"] == case_id) & (dynamics["criterion_id"] == criterion_id)
    ]
    if not case_dyn.empty:
        row = case_dyn.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Встреч", int(row["meetings_count"]))
        m2.metric("Средний балл", f"{row['mean_score']:.2f}")
        m3.metric("Наклон", f"{row['slope']:.3f}")
        m4.metric("Тренд", str(row["trend_status"]))

    series = scores[
        (scores["id_кейса"] == case_id) & (scores["criterion_id"] == criterion_id)
    ].sort_values("дата_встречи")
    fig = px.line(
        series,
        x="дата_встречи",
        y="value",
        markers=True,
        title=f"Динамика: {labels.get(criterion_id, criterion_id)}",
        range_y=[-0.1, 1.1],
    )
    fig.update_yaxes(dtick=1, title="Оценка (0/1)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Цитаты из отчётов (evidence)")
    evidence = series[["дата_встречи", "value", "evidence"]].copy()
    evidence["дата_встречи"] = pd.to_datetime(evidence["дата_встречи"]).dt.strftime("%Y-%m-%d")
    st.dataframe(evidence, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Анализ отчётов наставников")
    st.caption("Фонд «Волонтёры в помощь детям-сиротам» — пайплайн оценки встреч")

    criteria_df = _criteria_table()
    labels = dict(zip(criteria_df["id"], criteria_df["name"]))

    st.sidebar.header("Данные")
    uploaded = st.sidebar.file_uploader("Excel-файл с отчётами", type=["xlsx"])
    llm_available = gigachat_configured()
    use_llm = st.sidebar.toggle(
        "GigaChat (LLM)",
        value=llm_available,
        disabled=not llm_available,
        help="Нужны GIGACHAT_CREDENTIALS в .env. Иначе используется демо-эвристика.",
    )
    if not llm_available:
        st.sidebar.warning("GigaChat не настроен — анализ пойдёт в демо-режиме.")

    run = st.sidebar.button("Запустить анализ", type="primary", disabled=uploaded is None)
    if run and uploaded is not None:
        with st.spinner("Оцениваем отчёты…"):
            _run_analysis(uploaded, use_llm=use_llm and llm_available)

    if "export_meetings" in st.session_state:
        st.sidebar.markdown("#### Выгрузка в 1С")
        st.sidebar.download_button(
            "meetings_scores.csv",
            data=st.session_state["export_meetings"],
            file_name="meetings_scores.csv",
            mime="text/csv",
        )
        st.sidebar.download_button(
            "dynamics_summary.csv",
            data=st.session_state["export_dynamics"],
            file_name="dynamics_summary.csv",
            mime="text/csv",
        )

    with st.expander("Критерии оценки (criteria.yaml)"):
        st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    scores = st.session_state.get("scores")
    dynamics = st.session_state.get("dynamics")
    if scores is None or scores.empty:
        st.info("Загрузите Excel и нажмите «Запустить анализ».")
        st.markdown(
            "Ожидаемые колонки: `id_кейса`, `id_волонтера`, `дата_встречи`, `текст_отчета`."
        )
        return

    tab_macro, tab_micro = st.tabs(["Макро", "Микро"])
    with tab_macro:
        _macro_tab(scores, dynamics, labels)
    with tab_micro:
        _micro_tab(scores, dynamics, labels)


if __name__ == "__main__":
    main()
