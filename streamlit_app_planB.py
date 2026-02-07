import re
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Parsing helpers
# -----------------------------
def split_patterns_cell(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [s.strip() for s in re.split(r"[;,]", str(x)) if s.strip()]

# -----------------------------
# Scoring helpers
# -----------------------------
def add_raw_and_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    z = df["z_score"]
    w = df["weight"]
    d = df["Principal_direction"]
    p = df["Power"].astype(str)

    raw = np.zeros(len(df), dtype=float)

    mask_down = (d == "down")
    mask_up = (d == "up")

    if mask_down.any():
        zd = z[mask_down]
        wd = w[mask_down]
        raw[mask_down] = np.select(
            [zd > -1, (zd <= -1) & (zd >= -2), zd < -2],
            [0, 1 * wd, 2 * wd],
            default=0
        )

    if mask_up.any():
        zu = z[mask_up]
        wu = w[mask_up]
        raw[mask_up] = np.select(
            [zu < 1, (zu >= 1) & (zu <= 2), zu > 2],
            [0, 1 * wu, 2 * wu],
            default=0
        )

    out = df.copy()
    out["raw_score"] = raw

    boost = (
        ((p == "z≥2") & (z >= 2)) |
        ((p == "z≤-2") & (z <= -2)) |
        ((p == "|z|≥2") & (z.abs() >= 2))
    )
    out["final_score"] = np.where(boost, out["raw_score"] * 2, out["raw_score"])
    return out

def add_pattern_scores(df_scores: pd.DataFrame,
                       df_patterns: pd.DataFrame,
                       pattern_col_scores: str = "Pattern",
                       final_col: str = "final_score",
                       pattern_col_patterns: str = "Код паттерна") -> pd.DataFrame:
    pat_score = (
        df_scores.groupby(pattern_col_scores, dropna=False)[final_col]
        .sum()
        .clip(upper=10)
        .rsub(10)
        .rename("pattern_score")
    )

    status = pd.cut(
        pat_score,
        bins=[-np.inf, 4, 7, np.inf],
        labels=["лимитирующий", "потенциал-лимитирующий", "контекстный"],
        right=True
    ).astype(str).rename("status")

    out = df_patterns.merge(
        pd.concat([pat_score, status], axis=1)
          .reset_index()
          .rename(columns={pattern_col_scores: pattern_col_patterns}),
        on=pattern_col_patterns,
        how="left"
    )
    return out

def add_domain_scores(df_patterns: pd.DataFrame,
                      domain_col: str = "Код домена",
                      pattern_score_col: str = "pattern_score",
                      status_col: str = "status") -> pd.DataFrame:
    W = df_patterns[status_col].map({
        "лимитирующий": 1.0,
        "потенциал-лимитирующий": 0.7,
        "контекстный": 0.4
    })

    tmp = df_patterns.copy()
    tmp["_W_"] = W
    tmp["_num_"] = tmp[pattern_score_col] * tmp["_W_"]

    dom_score = (tmp.groupby(domain_col)["_num_"].sum()
                 / tmp.groupby(domain_col)["_W_"].sum())
    dom_score.name = "domain_score"
    return dom_score.reset_index()

def safe_mean(vals):
    vals = [v for v in vals if pd.notna(v)]
    return np.mean(vals) if len(vals) else np.nan

def safe_min(vals):
    vals = [v for v in vals if pd.notna(v)]
    return np.min(vals) if len(vals) else np.nan

def phenotype_score_from_patterns(pattern_score_map: dict, must_list, support_list, context_list):
    must_vals = [pattern_score_map.get(p, np.nan) for p in must_list]
    sup_vals = [pattern_score_map.get(p, np.nan) for p in support_list]
    ctx_vals = [pattern_score_map.get(p, np.nan) for p in context_list]

    must = safe_min(must_vals) if len(must_list) else np.nan
    sup = safe_mean(sup_vals) if len(support_list) else np.nan
    ctx = safe_mean(ctx_vals) if len(context_list) else np.nan

    # перераспределение весов, если support/context отсутствуют
    w_must = 0.5 + (0.3 if pd.isna(sup) else 0) + (0.2 if pd.isna(ctx) else 0)
    score = w_must * must + (0 if pd.isna(sup) else 0.3 * sup) + (0 if pd.isna(ctx) else 0.2 * ctx)
    return score

def calc_indexes(ph_scores_by_code: dict):
    ph = np.array([ph_scores_by_code[f"PH-{i}"] for i in range(1, 10)], dtype=float)

    score_Ph1, score_Ph2, score_Ph3, score_Ph4, score_Ph5, score_Ph6, score_Ph7, score_Ph8, score_Ph9 = ph

    MHI_PH = 0.7 * np.nanmean(ph) + 0.3 * np.nanmin(ph)
    if np.nanmin(ph) < 4:
        MHI_PH = min(MHI_PH, 6.5)

    LTI = 10 - (0.3*(10-score_Ph1) + 0.3*(10-score_Ph2) + 0.25*(10-score_Ph5) + 0.15*(10-score_Ph8))
    if score_Ph5 < 4:
        LTI = min(LTI, 4.5)

    RSI = 10 - (0.35*(10-score_Ph4) + 0.3*(10-score_Ph9) + 0.2*(10-score_Ph5) + 0.1*(10-score_Ph4))
    if score_Ph4 < 4:
        RSI = min(RSI, 4.5)

    ITI = 10 - (0.4*(10-score_Ph1) + 0.3*(10-score_Ph9) + 0.2*(10-score_Ph5) + 0.1*(10-score_Ph4))

    ORI = (0.30*(10-score_Ph9) + 0.25*(10-score_Ph4) + 0.20*(10-score_Ph1) +
           0.15*(10-score_Ph5) + 0.10*(10-score_Ph7))

    ARI = 0.25*10 + 0.20*score_Ph7 + 0.15*score_Ph3 + 0.15*score_Ph4 + 0.15*score_Ph8 + 0.10*score_Ph5
    if min(score_Ph3, score_Ph4, score_Ph7) < 4:
        ARI = ARI * 0.8
    elif (score_Ph3 < 5) and (score_Ph7 < 5):
        ARI = ARI * 0.5

#    MM1 = 10 - np.nanstd(ph)
#    if MM1 < 0:
#        MM1 = 0

#    RPI = np.mean([RSI, ARI]) - LTI + 5

#    return {"MHI_PH": MHI_PH, "LTI": LTI, "RSI": RSI, "ITI": ITI, "ORI": ORI, "ARI": ARI, "MM1": MM1, "RPI": RPI}
    return {"IMH": MHI_PH, "LTI": LTI, "RSI": RSI, "IME": ITI, "ORI": ORI, "ARI": ARI}

# -----------------------------
# Styling without matplotlib (Plan B)
# -----------------------------
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def css_red_to_green(v, vmin=1.0, vmax=10.0):
    # Accept only numeric-like
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, str):
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    if vmax == vmin:
        t = 1.0
    else:
        t = _clamp01((v - vmin) / (vmax - vmin))
    r = int(round(255 * (1 - t)))
    g = int(round(255 * t))
    b = 0
    return f"background-color: rgb({r},{g},{b}); color: black;"

def css_ori(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, str):
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    if v < 4.5:
        return "background-color: rgb(0,200,0); color: black;"       # green
    if v <= 6.4:
        return "background-color: rgb(255,215,0); color: black;"     # yellow
    return "background-color: rgb(255,0,0); color: black;"           # red

def _numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def style_table_general(df: pd.DataFrame, vmin=1.0, vmax=10.0):
    num_cols = _numeric_cols(df)
    sty = df.style
    if num_cols:
        sty = sty.applymap(lambda x: css_red_to_green(x, vmin=vmin, vmax=vmax), subset=num_cols)
        sty = sty.format("{:.1f}", subset=num_cols)
    return sty

def style_indexes(df: pd.DataFrame):
    num_cols = _numeric_cols(df)
    sty = df.style
    if num_cols:
        sty = sty.applymap(lambda x: css_red_to_green(x, vmin=0.0, vmax=10.0), subset=num_cols)
        sty = sty.format("{:.1f}", subset=num_cols)
    if "ORI" in df.columns:
        sty = sty.applymap(css_ori, subset=["ORI"])
        # ORI тоже форматируем (если вдруг стала object)
        try:
            sty = sty.format("{:.1f}", subset=["ORI"])
        except Exception:
            pass
    return sty

# -----------------------------
# Load definitions from uploaded xlsx (cached)
# -----------------------------
@st.cache_data
def load_definitions_from_bytes(calc_bytes: bytes):
    xf = pd.ExcelFile(BytesIO(calc_bytes))
    df_metab = pd.read_excel(xf, sheet_name="Metab_calc")
    df_patterns_template = pd.read_excel(xf, sheet_name="Pattern_calc")
    df_pheno = pd.read_excel(xf, sheet_name="Phenotype_calc")

    ph_defs = {}
    for _, r in df_pheno.iterrows():
        ph_code = r["PH_code"]
        ph_defs[ph_code] = {
            "must": split_patterns_cell(r.get("Must patterns", "")),
            "support": split_patterns_cell(r.get("Support patterns", "")),
            "context": split_patterns_cell(r.get("Context patterns", "")),
        }
    return df_metab, df_patterns_template, ph_defs

def compute_for_one_sample(sample_row: pd.Series,
                           df_metab_calc: pd.DataFrame,
                           df_patterns_template: pd.DataFrame,
                           ph_defs: dict):
    marker_cols = [c for c in sample_row.index if c not in ("Код", "Группа")]
    df_test = pd.DataFrame({"Metabolite": marker_cols, "z_score": sample_row[marker_cols].values})

    df_calc = df_metab_calc.drop(columns=["z-score"], errors="ignore").copy()
    df_calc = df_calc.merge(df_test, on="Metabolite", how="left")
    df_calc = add_raw_and_final_scores(df_calc)

    df_patterns = add_pattern_scores(df_calc, df_patterns_template.copy())
    df_domains = add_domain_scores(df_patterns)

    pat_map = dict(zip(df_patterns["Код паттерна"], df_patterns["pattern_score"]))

    ph_scores = {}
    for ph_code in [f"PH-{i}" for i in range(1, 10)]:
        defs = ph_defs.get(ph_code, {"must": [], "support": [], "context": []})
        ph_scores[ph_code] = phenotype_score_from_patterns(pat_map, defs["must"], defs["support"], defs["context"])

    idx = calc_indexes(ph_scores)

    dom_series = df_domains.set_index("Код домена")["domain_score"]
    return dom_series, pd.Series(ph_scores), pd.Series(idx), df_patterns

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="MetaboScan: Domains / Phenotypes / Indexes", layout="wide")
st.title("MetaboScan — расчёт доменов, фенотипов и индексов (plasma)")

st.sidebar.header("Файлы")
calc_file = st.sidebar.file_uploader("Загрузите calculations_new.xlsx", type=["xlsx"])
test_file = st.file_uploader("Загрузите test_new.xlsx (образцы строками)", type=["xlsx"])

if calc_file is None:
    st.info("Сначала загрузите calculations_new.xlsx (файл с правилами расчёта).")
    st.stop()

if test_file is None:
    st.info("Загрузите test_new.xlsx (данные образцов), чтобы увидеть расчёты.")
    st.stop()

df_in = pd.read_excel(test_file)

id_col = "Код" if "Код" in df_in.columns else None
if id_col is None:
    st.warning("В файле нет колонки 'Код'. В качестве идентификатора будут использованы номера строк.")
    ids = [f"sample_{i}" for i in range(len(df_in))]
else:
    ids = df_in[id_col].astype(str).fillna("").tolist()

df_metab_calc, df_patterns_template, ph_defs = load_definitions_from_bytes(calc_file.getvalue())

domain_rows, phen_rows, index_rows = [], [], []
patterns_by_sample = {}

for i, (_, row) in enumerate(df_in.iterrows()):
    sample_id = ids[i] if i < len(ids) else f"sample_{i}"
    dom_s, ph_s, idx_s, df_pat = compute_for_one_sample(row, df_metab_calc, df_patterns_template, ph_defs)

    dom_s.name = sample_id
    ph_s.name = sample_id
    idx_s.name = sample_id

    domain_rows.append(dom_s)
    phen_rows.append(ph_s)
    index_rows.append(idx_s)

    patterns_by_sample[sample_id] = df_pat

df_domains_out = pd.DataFrame(domain_rows)
df_phenotypes_out = pd.DataFrame(phen_rows)
df_indexes_out = pd.DataFrame(index_rows)

# -----------------------------
# Output tables with coloring (no matplotlib) + 1 decimal
# -----------------------------






st.subheader("Индексы")
st.dataframe(style_indexes(df_indexes_out), use_container_width=True)

st.subheader("Фенотипы")
st.dataframe(style_table_general(df_phenotypes_out, vmin=1.0, vmax=10.0), use_container_width=True)

st.subheader("Функциональные оси")
st.dataframe(style_table_general(df_domains_out, vmin=1.0, vmax=10.0), use_container_width=True)
# -----------------------------
# Domain → patterns drilldown (fallback UI)
# -----------------------------
st.divider()
st.subheader("Детализация: домен → паттерны (по выбранному образцу)")

c1, c2 = st.columns([1, 1])
with c1:
    sel_sample = st.selectbox("Образец", options=list(patterns_by_sample.keys()))
with c2:
    available_domains = sorted(df_patterns_template["Код домена"].dropna().unique().tolist()) if "Код домена" in df_patterns_template.columns else []
    sel_domain = st.selectbox("Домен", options=available_domains)

df_pat = patterns_by_sample.get(sel_sample)
if df_pat is not None and sel_domain is not None:
    if "Код домена" in df_pat.columns:
        sub = df_pat[df_pat["Код домена"] == sel_domain].copy()
    else:
        sub = df_pat.copy()

    show_cols = [c for c in ["Код домена", "Код паттерна", "pattern_score", "status"] if c in sub.columns]
    if len(show_cols) == 0:
        st.warning("В таблице паттернов не найдены ожидаемые колонки.")
    else:
        sub = sub[show_cols].sort_values(["Код домена", "Код паттерна"] if "Код домена" in show_cols else ["Код паттерна"])
        view = sub.set_index("Код паттерна") if "Код паттерна" in sub.columns else sub
        st.dataframe(style_table_general(view, vmin=1.0, vmax=10.0), use_container_width=True)
