
def make_design(df: pd.DataFrame, regressors: list[str]) -> pd.DataFrame:
    """Create OLS design matrix with intercept, regressors, city FE, and year FE."""
    parts = [pd.Series(1.0, index=df.index, name="Intercept")]
    for col in regressors:
        parts.append(pd.to_numeric(df[col], errors="coerce").astype(float).rename(col))

    city_fe = pd.get_dummies(df["城市"], prefix="城市", drop_first=True, dtype=float)
    year_fe = pd.get_dummies(df["年份"], prefix="年份", drop_first=True, dtype=float)
    return pd.concat(parts + [city_fe, year_fe], axis=1)


def fit_ols_cluster(
    df: pd.DataFrame,
    y_col: str,
    regressors: list[str],
    cluster_col: str = "城市",
) -> dict:
    """
    OLS with city and year fixed effects.
    Standard errors are clustered at city level.
    """
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
    finite_correction = (g / (g - 1)) * ((n - 1) / (n - k))
    cov_cluster = finite_correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov_cluster), 0))
    t_values = beta / se

    if stats is not None:
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
        "n": n,
        "k": k,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "table": table,
    }


def keep_core_rows(model: dict, variables: list[str]) -> pd.DataFrame:
    table = model["table"].set_index("变量")
    return table.loc[variables].reset_index()


def main() -> None:
    did = read_excel(DID_FILE)
    mediator = read_excel(MEDIATOR_FILE)
    controls = read_excel(CONTROL_FILE)
    resilience = read_excel(RESILIENCE_FILE)

    panel = did.merge(mediator, on=["城市", "年份"], how="inner", validate="one_to_one")
    panel = panel.merge(controls, on=["城市", "年份"], how="inner", validate="one_to_one")
    panel = panel.merge(resilience, on=["城市", "年份"], how="inner", validate="one_to_one")
    panel = panel.dropna().copy()
    panel["年份"] = panel["年份"].astype(int)

    stage1 = fit_ols_cluster(panel, "产业结构", ["DID"] + CONTROLS)
    stage2 = fit_ols_cluster(panel, "Eco_Resilience", ["DID", "产业结构"] + CONTROLS)

    result = {
        "sample": {
            "n_obs": int(len(panel)),
            "n_cities": int(panel["城市"].nunique()),
            "year_min": int(panel["年份"].min()),
            "year_max": int(panel["年份"].max()),
        },
        "stage1": {
            "model": "产业结构 = DID + 控制变量 + 城市固定效应 + 年份固定效应",
            "r2": stage1["r2"],
            "adj_r2": stage1["adj_r2"],
            "coefficients": keep_core_rows(stage1, ["DID"] + CONTROLS).to_dict(orient="records"),
        },
        "stage2": {
            "model": "Eco_Resilience = DID + 产业结构 + 控制变量 + 城市固定效应 + 年份固定效应",
            "r2": stage2["r2"],
            "adj_r2": stage2["adj_r2"],
            "coefficients": keep_core_rows(stage2, ["DID", "产业结构"] + CONTROLS).to_dict(orient="records"),
        },
    }

    json_path = OUTPUT_DIR / "analysis_mediation_regression_results.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    xlsx_path = OUTPUT_DIR / "analysis_mediation_regression_results.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        panel.to_excel(writer, sheet_name="合并面板数据", index=False)
        keep_core_rows(stage1, ["DID"] + CONTROLS).to_excel(writer, sheet_name="第一阶段结果", index=False)
        keep_core_rows(stage2, ["DID", "产业结构"] + CONTROLS).to_excel(writer, sheet_name="第二阶段结果", index=False)
        pd.DataFrame(
            [
                ["样本量", result["sample"]["n_obs"]],
                ["城市数", result["sample"]["n_cities"]],
                ["年份范围", f"{result['sample']['year_min']}-{result['sample']['year_max']}"],
                ["第一阶段R2", stage1["r2"]],
                ["第一阶段调整R2", stage1["adj_r2"]],
                ["第二阶段R2", stage2["r2"]],
                ["第二阶段调整R2", stage2["adj_r2"]],
                ["标准误", "按城市聚类"],
                ["固定效应", "城市固定效应、年份固定效应"],
            ],
            columns=["项目", "结果"],
        ).to_excel(writer, sheet_name="结果摘要", index=False)

    print("样本量:", result["sample"]["n_obs"])
    print("城市数:", result["sample"]["n_cities"])
    print("年份:", result["sample"]["year_min"], "-", result["sample"]["year_max"])
    print("\n第一阶段 R2:", round(stage1["r2"], 6), "调整R2:", round(stage1["adj_r2"], 6))
    print(keep_core_rows(stage1, ["DID"] + CONTROLS).to_string(index=False))
    print("\n第二阶段 R2:", round(stage2["r2"], 6), "调整R2:", round(stage2["adj_r2"], 6))
    print(keep_core_rows(stage2, ["DID", "产业结构"] + CONTROLS).to_string(index=False))
    print("\n结果已保存：")
    print(json_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()