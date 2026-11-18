"""Select the exact plot inputs from the executed notebook."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


def export(namespace, destination):
    ns = namespace
    ages = ns["AGES"]
    emotions = ns["EMOTIONS"]
    age_means = ns["age_means"].reindex(ages)
    age_means.index.name = "Возраст"
    gender_order = ["Женщины", "Мужчины"]
    gender = ns["gender_means"].reindex(gender_order)
    ext = ns["ext"]

    def with_age(frame):
        frame = frame.reindex(ages).copy()
        frame.insert(0, "Возраст (начало интервала)", [int(age.split("–")[0]) for age in ages])
        return frame

    length = ns["length_bins"]
    rates = length[length["n_dreams"] >= 30].pivot(
        index="age_group", columns="length_bin", values="rate_per_1000"
    )
    rates.columns = [f"{label} токенов" for label in rates]
    null_slopes = ns["null_slopes"]
    counts, edges = np.histogram(null_slopes, bins=25)
    histogram = pd.DataFrame({
        "Интервал": [f"{left:.3f}…{right:.3f}" for left, right in zip(edges[:-1], edges[1:])],
        "Симуляции": counts,
    })
    miami = ns["plot_data"][["prevalence", "ci_low", "ci_high"]] * 100
    barb = ns["plot_b"][["diff_b_minus_a", "boot_ci_low", "boot_ci_high"]]
    intervals = ["Оценка", "Нижняя граница 95% ДИ", "Верхняя граница 95% ДИ"]
    miami.columns = intervals
    barb.columns = intervals
    miami.index.name = "Серия"
    barb.index.name = "Категория NRC"

    mde = ns["mde_df"].copy()
    mde["Базовая доля, %"] = mde["base_rate"] * 100
    mde["mde_pp"] = mde["mde"] * 100

    tables = {
        "age_emotions": age_means[emotions].T.rename_axis("Категория NRC").reset_index(),
        "age_vader": with_age(age_means[["vader"]].rename(columns={"vader": "VADER compound"})),
        "age_length": with_age(age_means[["token_count"]].rename(columns={"token_count": "Токенов"})),
        "age_mattr": with_age(age_means[["mattr_50"]].rename(columns={"mattr_50": "MATTR-50"})),
        "age_topics": ns["topic_by_age"].reindex(ages).T.rename_axis("Тема").reset_index(),
        "gender_length": gender[["token_count"]].rename(columns={"token_count": "Токенов"})
        .rename_axis("Группа").reset_index(),
        "gender_emotions": gender[emotions].T.rename_axis("Категория NRC").reset_index(),
        "words_common": ns["common"][["word", "female", "male"]]
        .rename(columns={"word": "Слово", "female": "Женщины", "male": "Мужчины"}),
        "words_female": ns["female_words"][["word", "difference"]]
        .rename(columns={"word": "Слово", "difference": "Женщины − мужчины"}),
        "words_male": ns["male_words"][["word", "difference"]].assign(
            difference=lambda frame: -frame["difference"]
        ).rename(columns={"word": "Слово", "difference": "Мужчины − женщины"}),
        "anger_length": with_age(rates),
        "anger_null": histogram,
        "miami": miami.reset_index(),
        "barb": barb.reset_index(),
        "school": with_age(ns["school_by_age"].rename(columns={
            "school_topic_base": "С именами", "school_topic_no_names": "Без имён",
        })),
    }
    for series in ("norms-f", "norms-m"):
        matrix = ext.odds_ratio_matrix(ns["cooc_tables"][series], "mh_odds_ratio", emotions)
        tables[series] = matrix.rename_axis("Категория NRC").reset_index()
    for effect, frame in mde.groupby("deff", sort=True):
        tables[f"mde_{effect:g}"] = frame.pivot(
            index="Базовая доля, %", columns="design", values="mde_pp",
        ).reset_index()

    snapshot = {
        "schema": 1,
        "source": "notebooks/DreamPoly.ipynb",
        "observed_slope": float(ns["observed_slope"]),
        "tables": {
            key: {
                **json.loads(frame.to_json(orient="split", index=False, double_precision=15)),
                "types": ["number" if pd.api.types.is_numeric_dtype(dtype) else "string"
                          for dtype in frame.dtypes],
            }
            for key, frame in tables.items()
        },
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(snapshot, file, ensure_ascii=False, allow_nan=False, indent=2)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
