from pathlib import Path
import json
import math
import pandas as pd
import numpy as np

try:
    from scipy import stats
except Exception:
    stats = None


RAW_ECOLOGY_FILE = Path(r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\城市生态韧性\城市生态韧性原始数据.xlsx")
DID_FILE = Path(r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\DID.xlsx")
CONTROL_FILE = Path(r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\控制变量.xlsx")
RESILIENCE_FILE = Path(r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\城市生态韧性\熵权法_城市生态韧性.xlsx")

OUTPUT_DIR = Path(r"C:\Users\21026\Documents\Codex\2026-05-23\files-mentioned-by-the-user-did\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRESSURE_COLS = [
    "人均工业废水排放量",
    "人均氧化硫排放量",
    "人均工业烟尘排放量",
    "人均碳排放量",
    "PM2.5年平均浓度",
]
CONTROLS = ["人口规模", "经济发展水平", "对外开放水平", "城镇化率", "医疗卫生水平"]


def normal_two_sided_p(z: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def read_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")
    df["城市"] = df["城市"].astype(str).str.strip()
    df["年份"] = pd.to_numeric(df["年份"], errors="coerce").astype("Int64")
    return df


def minmax_positive(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    xmin, xmax = x.min(), x.max()
    if pd.isna(xmin) or pd.isna(xmax) or xmax == xmin:
        return pd.Series(0.0, index=series.index)
    return (x - xmin) / (xmax - xmin)


def entropy_weights(normalized: pd.DataFrame) -> pd.Series:
    x = normalized.astype(float) + 1e-12
    p = x.div(x.sum(axis=0), axis=1)
    entropy = -(p * np.log(p)).sum(axis=0) / np.log(len(x))
    redundancy = 1 - entropy
    if float(redundancy.sum()) == 0:
        return pd.Series(1 / len(x.columns), index=x.columns)
    return redundancy / redundancy.sum()


def make_design(df: pd.DataFrame, regressors: list[str]) -> pd.DataFrame:
    parts = [pd.Series(1.0, index=df.index, name="Intercept")]
    for col in regressors:
        parts.append(pd.to_numeric(df[col], errors="coerce").astype(float).rename(col))
    city_fe = pd.get_dummies(df["城市"], prefix="城市", drop_first=True, dtype=float)
    year_fe = pd.get_dummies(df["年份"], prefix="年份", drop_first=True, dtype=float)
    return pd.concat(parts + [city_fe, year_fe], axis=1)


def fit_ols_cluster(df: pd.DataFrame, y_col: str, regressors: list[str], cluster_col: str = "城市") -> dict:
    y = pd.to_numeric(df[y_col], errors="coerce").astype(float).to_numpy()
    x_df = make_design(df, regressors)
    x = x_df.to_numpy(dtype=float)
    n, k = x.shape

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    fitted = x @ beta
    resid = y - fitted

    tss = float(np.sum((y - y.mean()) ** 2))
    rss = float(np.sum(resid ** 2))
    r2 = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    clusters = df[cluster_col].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    meat = np.zeros((k, k), dtype=float)
    for group in unique_clusters:
        idx = clusters == group
        score = x[idx, :].T @ resid[idx]
        meat += np.outer(score, score)

    g = len(unique_clusters)
    finite = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    cov = finite * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_values = beta / se

    if stats is not None and g > 1:
        p_values = 2 * stats.t.sf(np.abs(t_values), df=g - 1)
    else:
        p_values = np.array([normal_two_sided_p(t) for t in t_values])

    table = pd.DataFrame(
        {
            "变量": x_df.columns,
            "系数": beta,
            "聚类稳健标准误": se,
            "t值": t_values,
            "p值": p_values,
        }
    )
    table["显著性"] = table["p值"].map(stars)
    return {
        "n": int(n),
        "n_cities": int(df["城市"].nunique()),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "table": table,
    }


def core_rows(model: dict, variables: list[str]) -> pd.DataFrame:
    table = model["table"].set_index("变量")
    return table.loc[variables].reset_index()


def build_pressure_groups() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    raw = read_excel(RAW_ECOLOGY_FILE)
    pressure = raw[["城市", "年份"] + PRESSURE_COLS].dropna().copy()
    normalized = pressure[PRESSURE_COLS].apply(minmax_positive)
    weights = entropy_weights(normalized)
    pressure["生态压力指数"] = normalized.mul(weights, axis=1).sum(axis=1)

    pre2020 = pressure[pressure["年份"] < 2020].copy()
    city_pressure = (
        pre2020.groupby("城市", as_index=False)["生态压力指数"]
        .mean()
        .rename(columns={"生态压力指数": "2020年前平均生态压力指数"})
    )
    threshold = float(city_pressure["2020年前平均生态压力指数"].median())
    city_pressure["高压力组"] = (city_pressure["2020年前平均生态压力指数"] > threshold).astype(int)
    city_pressure["压力组"] = np.where(city_pressure["高压力组"] == 1, "高压力组", "低压力组")
    weight_table = pd.DataFrame({"压力指标": weights.index, "熵权": weights.values})
    return pressure, city_pressure, weight_table, threshold


def main() -> None:
    pressure_index, city_pressure, weight_table, threshold = build_pressure_groups()
    did = read_excel(DID_FILE)
    controls = read_excel(CONTROL_FILE)
    resilience = read_excel(RESILIENCE_FILE)

    panel = did.merge(controls, on=["城市", "年份"], how="inner", validate="one_to_one")
    panel = panel.merge(resilience, on=["城市", "年份"], how="inner", validate="one_to_one")
    panel = panel.merge(city_pressure, on="城市", how="inner", validate="many_to_one")
    panel = panel.dropna().copy()
    panel["年份"] = panel["年份"].astype(int)

    regressors = ["DID"] + CONTROLS
    all_tables = []
    outputs = {}
    for group_name in ["低压力组", "高压力组"]:
        sub = panel[panel["压力组"] == group_name].copy()
        model = fit_ols_cluster(sub, "Eco_Resilience", regressors)
        core = core_rows(model, ["Intercept"] + regressors)
        core.insert(0, "压力组", group_name)
        core["样本量"] = model["n"]
        core["城市数"] = model["n_cities"]
        core["R2"] = model["r2"]
        core["调整R2"] = model["adj_r2"]
        all_tables.append(core)
        outputs[group_name] = {
            "n_obs": model["n"],
            "n_cities": model["n_cities"],
            "r2": model["r2"],
            "adj_r2": model["adj_r2"],
            "coefficients": core.to_dict(orient="records"),
        }

    results = pd.concat(all_tables, ignore_index=True)
    did_compare = results[results["变量"] == "DID"].copy()
    key_compare = results[results["变量"].isin(["Intercept", "DID"])].copy()
    low_coef = float(did_compare.loc[did_compare["压力组"] == "低压力组", "系数"].iloc[0])
    high_coef = float(did_compare.loc[did_compare["压力组"] == "高压力组", "系数"].iloc[0])
    comparison = pd.DataFrame(
        [
            ["低压力组DID系数", low_coef],
            ["高压力组DID系数", high_coef],
            ["系数差值：高压力-低压力", high_coef - low_coef],
            ["系数倍数：高压力/低压力", high_coef / low_coef if low_coef != 0 else np.nan],
        ],
        columns=["项目", "结果"],
    )

    sample_summary = panel.groupby("压力组").agg(
        样本量=("城市", "size"),
        城市数=("城市", "nunique"),
        DID处理观测数=("DID", "sum"),
        平均压力指数=("2020年前平均生态压力指数", "mean"),
        最小压力指数=("2020年前平均生态压力指数", "min"),
        最大压力指数=("2020年前平均生态压力指数", "max"),
    ).reset_index()

    summary = {
        "method": {
            "pressure_indicators": PRESSURE_COLS,
            "normalization": "正向标准化，数值越大表示生态压力越高",
            "weighting": "熵权法",
            "grouping": "按城市2020年前平均生态压力指数的中位数分组；高于中位数为高压力组，其余为低压力组",
            "median_pressure": threshold,
        },
        "model": "Eco_Resilience = DID + 控制变量 + 城市固定效应 + 年份固定效应，按压力组分样本估计",
        "weights": weight_table.to_dict(orient="records"),
        "sample_summary": sample_summary.to_dict(orient="records"),
        "did_comparison": comparison.to_dict(orient="records"),
        "results": outputs,
    }

    json_path = OUTPUT_DIR / "heterogeneity_pressure_group_regression_results.json"
    xlsx_path = OUTPUT_DIR / "heterogeneity_pressure_group_regression_results.xlsx"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame([summary["method"]]).to_excel(writer, sheet_name="方法说明", index=False)
        weight_table.to_excel(writer, sheet_name="熵权", index=False)
        sample_summary.to_excel(writer, sheet_name="样本摘要", index=False)
        city_pressure.sort_values("2020年前平均生态压力指数", ascending=False).to_excel(writer, sheet_name="城市压力分组", index=False)
        results.to_excel(writer, sheet_name="分组回归结果", index=False)
        did_compare.to_excel(writer, sheet_name="DID系数对比", index=False)
        key_compare.to_excel(writer, sheet_name="DID和常数项", index=False)
        comparison.to_excel(writer, sheet_name="系数大小比较", index=False)
        pressure_index.to_excel(writer, sheet_name="年度压力指数", index=False)
        panel.to_excel(writer, sheet_name="回归面板数据", index=False)

    print("熵权：")
    print(weight_table.to_string(index=False))
    print(f"\n分组阈值: {threshold:.6f}")
    print("\n样本摘要：")
    print(sample_summary.to_string(index=False))
    print("\nDID系数对比：")
    print(did_compare[["压力组", "系数", "聚类稳健标准误", "t值", "p值", "显著性", "样本量", "城市数", "R2", "调整R2"]].to_string(index=False))
    print("\nDID与常数项：")
    print(key_compare[["压力组", "变量", "系数", "聚类稳健标准误", "t值", "p值", "显著性"]].to_string(index=False))
    print("\n系数大小比较：")
    print(comparison.to_string(index=False))
    print("\n结果已保存：")
    print(json_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
