# -*- coding: utf-8 -*-
"""Обвязка секции 11: подготовка данных, запуск extended, сохранение csv/png."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from dreambank.analytics import extended as ext
from dreambank.analytics.config import IzzyAgeConfig
from dreambank.analytics.constants import EMOTIONS
from dreambank.analytics.features import attach_features, prepare_age_features, tokens
from dreambank.parsing.frames import DreamsFrame
from dreambank.parsing.html import ParsedArchive


# ---------------------------------------------------------------------------
# Подготовка данных (каждый раздел 11.x пересчитывает своё, как в develop
# уже делают IzzyEmotionsByAge / IzzyLengthVocabulary / IzzyTopics)
# ---------------------------------------------------------------------------

def _dreams_frame(archive: ParsedArchive) -> pd.DataFrame:
    return pd.DataFrame(archive.dreams).reset_index(drop=True)


AGGRESSION_SERIES = ["norms-f", "norms-m", "college-f", "college-m",
                     "miami-home", "miami-lab"]
NORMS_SERIES = ["norms-f", "norms-m"]
BARB_SERIES = ["b", "b2"]


def _prepare_aggression(archive: ParsedArchive) -> pd.DataFrame:
    dreams = _dreams_frame(archive)
    data = dreams.loc[
        dreams["series_id"].isin(AGGRESSION_SERIES)
        & dreams["language"].eq("en")
    ].copy()
    if data.empty:
        raise ValueError("No aggression series dreams in archive")
    data["token_count"] = data["text"].apply(lambda t: len(tokens(t)))
    data, counts_broad = ext.add_marker_columns(
        data, ext.AGGRESSION_PATTERNS_BROAD, "aggr_broad")
    data, counts_strict = ext.add_marker_columns(
        data, ext.AGGRESSION_PATTERNS_STRICT, "aggr_strict")
    return data, counts_broad, counts_strict


def _prepare_norms(archive: ParsedArchive) -> pd.DataFrame:
    dreams = _dreams_frame(archive)
    data = dreams.loc[
        dreams["series_id"].isin(NORMS_SERIES) & dreams["language"].eq("en")
    ].copy()
    if data.empty:
        raise ValueError("No norms series dreams in archive")
    data = pd.DataFrame(attach_features(DreamsFrame.build(data)))
    data = ext.add_emotion_presence(data, EMOTIONS)
    data["length_stratum"] = ext.length_strata(data["token_count"])
    return data


def _prepare_barb(archive: ParsedArchive) -> pd.DataFrame:
    dreams = _dreams_frame(archive)
    data = dreams.loc[dreams["series_id"].isin(BARB_SERIES)].copy()
    if data.empty:
        raise ValueError("No Barb Sanders series dreams in archive")
    data = pd.DataFrame(attach_features(DreamsFrame.build(data)))
    data["dream_number"] = pd.to_numeric(data["dream_number_raw"], errors="coerce")
    data = data.dropna(subset=["dream_number"]).sort_values(["series_id", "dream_number"])
    data = data[data["token_count"] >= 50].copy()
    return data


def _prepare_izzy(archive: ParsedArchive) -> pd.DataFrame:
    ages = IzzyAgeConfig().ages
    return pd.DataFrame(prepare_age_features(archive, ages))


def _ensure_dirs(output: Path, figures: Path) -> tuple[Path, Path]:
    ext_out = Path(output) / "extended"
    ext_fig = Path(figures) / "extended"
    ext_out.mkdir(parents=True, exist_ok=True)
    ext_fig.mkdir(parents=True, exist_ok=True)
    return ext_out, ext_fig


# ---------------------------------------------------------------------------
# 11.1 + 11.4 + 11.8 — лексические маркеры агрессии, Miami, MDE
# (в одном эксперименте, чтобы не считать aggression_data трижды)
# ---------------------------------------------------------------------------

def run_aggression(archive: ParsedArchive, output: Path, figures: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns  # noqa: F401  — для проверки, что зависимость стоит
    from scipy.stats import spearmanr  # noqa: F401  — не нужен, но напомнит, если расширите

    matplotlib.use("Agg", force=True)
    out, fig_dir = _ensure_dirs(output, figures)
    data, counts_broad, _ = _prepare_aggression(archive)

    # --- 11.1: маркеры по сериям (широкий и строгий) ---
    data.to_csv(out / "aggression_lexical_markers_by_dream.csv",
                index=False, encoding="utf-8-sig")

    summary_broad = ext.summarize_by_series(
        data, "aggr_broad_present", "aggr_broad_count").assign(dictionary="broad")
    summary_strict = ext.summarize_by_series(
        data, "aggr_strict_present", "aggr_strict_count").assign(dictionary="strict")
    pd.concat([summary_broad, summary_strict], ignore_index=True).to_csv(
        out / "aggression_lexical_markers_by_series.csv",
        index=False, encoding="utf-8-sig")

    counts_broad.sum().sort_values(ascending=False).rename("occurrences").rename_axis(
        "marker").reset_index().to_csv(
        out / "aggression_marker_frequency.csv", index=False, encoding="utf-8-sig")

    # KWIC-аудит словаря
    ext.kwic_sample(data, ext.AGGRESSION_PATTERNS_BROAD, n_per_marker=20, seed=ext.RANDOM_SEED) \
        .to_csv(out / "aggression_kwic_audit.csv", index=False, encoding="utf-8-sig")

    # --- 11.1: половые сравнения norms и college ---
    sex_rows = []
    for dictionary, prefix in [("broad", "aggr_broad"), ("strict", "aggr_strict")]:
        for series_a, series_b in [("norms-m", "norms-f"), ("college-m", "college-f")]:
            row = ext.compare_two_series(
                data, series_a, series_b, f"{prefix}_present", f"{prefix}_count")
            row["dictionary"] = dictionary
            sex_rows.append(row)
    sex_tests = pd.DataFrame(sex_rows)
    for column in ["fisher_p", "adj_or_p", "rr_p"]:
        sex_tests[column + "_bh"] = (
            sex_tests.groupby("dictionary")[column].transform(ext.adjust_pvalues))
    sex_tests.to_csv(out / "aggression_sex_comparisons.csv",
                     index=False, encoding="utf-8-sig")

    # --- 11.4: Miami home vs lab ---
    miami_rows = []
    for dictionary, prefix in [("broad", "aggr_broad"), ("strict", "aggr_strict")]:
        row = ext.compare_two_series(
            data, "miami-lab", "miami-home", f"{prefix}_present", f"{prefix}_count")
        row["dictionary"] = dictionary
        miami_rows.append(row)
    miami_tests = pd.DataFrame(miami_rows)
    miami_tests["fisher_p_bh"] = ext.adjust_pvalues(miami_tests["fisher_p"])
    miami_tests.to_csv(out / "miami_lexical_aggression_tests.csv",
                       index=False, encoding="utf-8-sig")

    # график Miami (строгий словарь)
    plot_data = summary_strict[summary_strict["series_id"].isin(
        ["miami-home", "miami-lab"])].set_index("series_id").loc[
        ["miami-home", "miami-lab"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(plot_data.index, plot_data["prevalence"] * 100,
           yerr=[(plot_data["prevalence"] - plot_data["ci_low"]) * 100,
                 (plot_data["ci_high"] - plot_data["prevalence"]) * 100],
           capsize=5)
    ax.set(ylabel="Доля снов хотя бы с одним маркером, %",
           title="Миами: строгий словарь маркеров агрессии")
    fig.tight_layout()
    fig.savefig(fig_dir / "miami_lexical_aggression_prevalence.png", dpi=150)
    plt.close(fig)

    # --- 11.8: MDE + Fisher power ---
    sizes = data["series_id"].value_counts()
    designs = {
        "college-f / college-m": (int(sizes["college-f"]), int(sizes["college-m"])),
        "norms-f / norms-m":     (int(sizes["norms-f"]),   int(sizes["norms-m"])),
        "miami-home / miami-lab": (int(sizes["miami-home"]), int(sizes["miami-lab"])),
    }
    mde_df = ext.mde_table(designs, [0.05, 0.10, 0.15, 0.20, 0.25],
                           power=0.8, alpha=0.05, deffs=(1.0, 1.5, 2.0))
    mde_df.to_csv(out / "minimum_detectable_difference.csv",
                  index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    for design, g in mde_df[mde_df["deff"] == 1.0].groupby("design"):
        ax.plot(g["base_rate"] * 100, g["mde"] * 100, marker="o", label=design)
    ax.set(xlabel="Базовая доля, %", ylabel="Минимальная обнаружимая разность, п. п.",
           title="Мощность 80%, α = 0,05, двусторонний критерий")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "minimum_detectable_difference.png", dpi=150)
    plt.close(fig)

    fisher_rows = []
    for design, (n1, n2) in designs.items():
        fisher_rows.append({
            "design": design, "n1": n1, "n2": n2,
            "power_fisher_13.9_vs_26.5": ext.simulate_fisher_power(
                n1, n2, 0.139, 0.265, n_sim=2000, seed=ext.RANDOM_SEED),
        })
    pd.DataFrame(fisher_rows).to_csv(
        out / "fisher_power_known_rates.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 11.2 — длина и лексика гнева
# ---------------------------------------------------------------------------

def run_length_anger(archive: ParsedArchive, output: Path, figures: Path) -> None:
    import matplotlib.pyplot as plt
    matplotlib.use("Agg", force=True)
    out, fig_dir = _ensure_dirs(output, figures)

    izzy = _prepare_izzy(archive)
    d = ext.prepare_length_data(izzy, emotion="anger", min_tokens=50,
                                max_tokens=500, block_size=20)

    res_hc3 = ext.fit_rate_model(d, "anger_count", age_ref="18–21")
    res_cluster = ext.fit_rate_model(d, "anger_count", age_ref="18–21", cluster="block")
    ext.rate_model_table(res_hc3).to_csv(
        out / "izzy_anger_rate_model_hc3.csv", encoding="utf-8-sig")
    ext.rate_model_table(res_cluster).to_csv(
        out / "izzy_anger_rate_model_cluster.csv", encoding="utf-8-sig")

    stat, p, df = ext.wald_block_test(res_hc3, "age_group")
    pd.DataFrame({"statistic": [stat], "p_value": [p], "df": [df]}).to_csv(
        out / "izzy_anger_wald.csv", index=False, encoding="utf-8-sig")

    # Контроль «механики»: распределение нулевых наклонов
    observed_slope, null_slopes = ext.simulate_null_logit_slope(
        d, "anger_count", n_sim=200, seed=ext.RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_slopes, bins=25, color="lightgray", edgecolor="gray")
    ax.axvline(observed_slope, color="crimson", label="данные")
    ax.set(xlabel="Наклон логита по log(токенов)", ylabel="Число симуляций",
           title="Постоянная частота против наблюдаемого наклона")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "izzy_length_null_slope.png", dpi=150)
    plt.close(fig)

    length_bins = ext.length_bin_table(d, "anger_count", edges=(50, 100, 200, 501))
    length_bins.to_csv(out / "izzy_anger_by_age_and_length_bin.csv",
                       index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    for lbin, g in length_bins.groupby("length_bin"):
        g = g[g["n_dreams"] >= 30]
        if g.empty:
            continue
        ax.plot(g["age_group"], g["rate_per_1000"], marker="o",
                label=f"{lbin} токенов")
    ax.set(xlabel="Возраст", ylabel="Слов anger на 1000 токенов",
           title="Лексика гнева по возрасту при одинаковой длине (группы от 30 снов)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "izzy_anger_by_age_length_bins.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11.3 — половины внутри возрастных папок Izzy
# ---------------------------------------------------------------------------

def run_izzy_stability(archive: ParsedArchive, output: Path, figures: Path) -> None:
    out, _ = _ensure_dirs(output, figures)
    izzy = _prepare_izzy(archive)
    izzy["dream_number"] = pd.to_numeric(izzy["dream_number_raw"], errors="coerce")
    izzy = izzy.dropna(subset=["dream_number", "age_group"])
    izzy = izzy[izzy["token_count"] >= 50].copy()

    metrics = ["anger", "fear", "sadness", "joy", "token_count", "mattr_50"]
    half_table = ext.compare_halves_by_group(
        izzy, "age_group", metrics, order_col="dream_number",
        min_n=20, seed=ext.RANDOM_SEED)
    half_table.to_csv(out / "izzy_halves_by_age.csv",
                      index=False, encoding="utf-8-sig")
    ext.summarize_half_differences(half_table).to_csv(
        out / "izzy_halves_summary.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 11.5 — совместная встречаемость NRC (norms-f/m)
# ---------------------------------------------------------------------------

def run_cooccurrence(archive: ParsedArchive, output: Path, figures: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.use("Agg", force=True)
    out, fig_dir = _ensure_dirs(output, figures)

    norms_data = _prepare_norms(archive)

    cooc_tables = {}
    for series_id, group in norms_data.groupby("series_id"):
        table = ext.cooccurrence_table(group, EMOTIONS)
        table.insert(0, "series_id", series_id)
        cooc_tables[series_id] = table
    cooc_df = pd.concat(cooc_tables.values(), ignore_index=True)
    cooc_df.to_csv(out / "nrc_emotion_cooccurrence.csv",
                   index=False, encoding="utf-8-sig")

    for series_id, table in cooc_tables.items():
        matrix = ext.odds_ratio_matrix(table, "mh_odds_ratio", EMOTIONS)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(np.log(matrix), annot=matrix.round(2), fmt="", cmap="coolwarm",
                    center=0, ax=ax, cbar_kws={"label": "log OR (цвет); OR (подпись)"})
        ax.set_title(f"NRC: OR Мантеля–Хэнзеля с поправкой на длину, {series_id}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"nrc_mh_or_{series_id}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 11.6 — Barb Sanders: b половины и b vs b2
# ---------------------------------------------------------------------------

def run_barb_sanders(archive: ParsedArchive, output: Path, figures: Path) -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    matplotlib.use("Agg", force=True)
    out, fig_dir = _ensure_dirs(output, figures)

    b_data = _prepare_barb(archive)
    b_only = b_data[b_data["series_id"] == "b"]
    b2_only = b_data[b_data["series_id"] == "b2"]
    metrics= [*EMOTIONS, "token_count"]

    b_first, b_second = ext.split_halves(b_only, "dream_number")
    b_halves = ext.compare_emotion_profiles(
        b_first, b_second, metrics, "first_half", "second_half", seed=ext.RANDOM_SEED)
    b_halves.to_csv(out / "barb_sanders_halves.csv", index=False, encoding="utf-8-sig")

    b_vs_b2 = ext.compare_emotion_profiles(
        b_only.sort_values("dream_number"), b2_only.sort_values("dream_number"),
        metrics, "b", "b2", seed=ext.RANDOM_SEED)
    b_vs_b2.to_csv(out / "barb_sanders_b_vs_b2.csv", index=False, encoding="utf-8-sig")

    # Контроль: Spearman профилей b и b2
    profile_b = b_only[list(EMOTIONS)].mean()
    profile_b2 = b2_only[list(EMOTIONS)].mean()
    pd.DataFrame({"profile": ["b_vs_b2"],
                  "spearman": [spearmanr(profile_b, profile_b2)[0]]}).to_csv(
        out / "barb_sanders_spearman.csv", index=False, encoding="utf-8-sig")

    plot_b = b_vs_b2[b_vs_b2["metric"].isin(EMOTIONS)].set_index("metric").loc[list(EMOTIONS)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(range(len(EMOTIONS)), plot_b["diff_b_minus_a"],
                yerr=[plot_b["diff_b_minus_a"] - plot_b["boot_ci_low"],
                      plot_b["boot_ci_high"] - plot_b["diff_b_minus_a"]],
                fmt="o", capsize=4)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=30)
    ax.set(ylabel="b2 − b, слов на 1000 токенов",
           title="Barb Sanders: разность профилей b2 и b (95% bootstrap по блокам)")
    fig.tight_layout()
    fig.savefig(fig_dir / "barb_sanders_b2_minus_b.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11.7 — имена и темы Izzy
# ---------------------------------------------------------------------------

NOT_NAMES = {
    "facebook", "runescape", "asian", "english", "sims", "warcraft", "pokemon",
    "supernatural", "technical", "development", "creek", "smallville", "firefly",
    "mcdonald", "midvale", "buffy", "tigger",
}


def run_names_topics(archive: ParsedArchive, output: Path, figures: Path) -> None:
    import matplotlib.pyplot as plt
    matplotlib.use("Agg", force=True)
    out, fig_dir = _ensure_dirs(output, figures)

    ages = IzzyAgeConfig().ages
    labels_ages = [band.label for band in ages]
    izzy = _prepare_izzy(archive)
    izzy_topics = izzy.sort_values(["series_id", "dream_number_raw"]).reset_index(drop=True)
    topic_texts = izzy_topics["text"].fillna("").tolist()

    names_table = ext.find_person_names(topic_texts, min_count=5, min_ratio=0.9)
    names_table.to_csv(out / "izzy_name_candidates.csv",
                       index=False, encoding="utf-8-sig")

    person_names = set(names_table["word"]) - NOT_NAMES

    fit_base = ext.fit_nmf_topics(topic_texts, extra_stop_words=(), seed=ext.RANDOM_SEED)
    fit_no = ext.fit_nmf_topics(topic_texts, extra_stop_words=person_names,
                                seed=ext.RANDOM_SEED)

    pd.DataFrame({
        "topic": range(1, 9),
        "base_terms": ext.topic_top_terms(fit_base),
        "no_names_terms": ext.topic_top_terms(fit_no),
    }).to_csv(out / "topics_base_vs_no_names.csv", index=False, encoding="utf-8-sig")

    ext.match_topics(fit_base, fit_no).to_csv(
        out / "topics_matching.csv", index=False, encoding="utf-8-sig")

    school_base = ext.topic_index_by_term(fit_base, "school")
    school_no = ext.topic_index_by_term(fit_no, "school")
    shares_base = ext.topic_shares_by_group(fit_base, izzy_topics["age_group"]) \
        .reindex(labels_ages)
    shares_no = ext.topic_shares_by_group(fit_no, izzy_topics["age_group"]) \
        .reindex(labels_ages)
    school_by_age = pd.DataFrame({
        "school_topic_base": shares_base[f"topic_{school_base + 1}"],
        "school_topic_no_names": shares_no[f"topic_{school_no + 1}"],
    })
    school_by_age.to_csv(out / "school_topic_share_by_age.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(labels_ages, school_by_age["school_topic_base"], marker="o", label="с именами")
    ax.plot(labels_ages, school_by_age["school_topic_no_names"], marker="o", label="без имён")
    ax.set(xlabel="Возраст", ylabel="Средний нормированный вес темы",
           title="Тема «школа» по возрастам: с именами и без")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "school_topic_with_without_names.png", dpi=150)
    plt.close(fig)