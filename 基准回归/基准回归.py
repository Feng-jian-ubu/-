import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

control_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\控制变量.xlsx"
resilience_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\城市生态韧性\熵权法_城市生态韧性.xlsx"
policy_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\DID.xlsx"


control = pd.read_excel(control_path)
resilience = pd.read_excel(resilience_path)
policy = pd.read_excel(policy_path)


resilience.rename(columns={"City": "城市", "Year": "年份"}, inplace=True)

df = control.merge(resilience, on=["城市", "年份"]) \
            .merge(policy, on=["城市", "年份"])

df.rename(columns={"DID": "Treat_Time"}, inplace=True)


y = "Eco_Resilience"

did_var = "Treat_Time"

controls = [
    "人口规模",
    "经济发展水平",
    "对外开放水平",
    "城镇化率",
    "医疗卫生水平"
]

all_vars = [y, did_var] + controls


var_label = {
    "Eco_Resilience": "生态韧性",
    "Treat_Time": "政策变量",
    "人口规模": "户籍人口（取对数）",
    "经济发展水平": "人均地区生产总值（取对数）",
    "对外开放水平": "实际利用外资额/地区生产总值",
    "城镇化率": "非农业人口/户籍人口",
    "医疗卫生水平": "每百人医院、卫生院床位"
}

formula1 = f"{y} ~ {did_var} + C(城市) + C(年份)"

formula2 = f"{y} ~ {did_var} + {' + '.join(controls)} + C(城市) + C(年份)"

formula3 = f"{y} ~ {did_var} + {' + '.join(controls)}"

results1 = smf.ols(formula1, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["城市"]}
)

results2 = smf.ols(formula2, data=df).fit(
    cov_type="HC1"
)

results3 = smf.ols(formula3, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["城市"]}
)

results4 = smf.ols(formula2, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["城市"]}
)

models = [results1, results2, results3, results4]


def star(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


variables = [
    did_var,
    "人口规模",
    "经济发展水平",
    "对外开放水平",
    "城镇化率",
    "医疗卫生水平",
    "Intercept"
]

var_names = [
    "Treat×Time",
    "人口规模",
    "经济发展水平",
    "对外开放水平",
    "城镇化率",
    "医疗卫生水平",
    "常数项"
]

table = {"变量": var_names.copy()}

for i in range(4):
    table[f"({i+1})"] = []

for var in variables:
    for i, m in enumerate(models):
        coef = m.params.get(var, float("nan"))
        se = m.bse.get(var, float("nan"))
        p = m.pvalues.get(var, 1)

        table[f"({i+1})"].append(
            f"{coef:.4f}{star(p)}\n({se:.4f})"
        )



table["变量"].extend([
    "城市固定效应",
    "年份固定效应",
    "样本量",
    "Adjusted R²"
])

city_fe = ["是", "是", "否", "是"]
year_fe = ["是", "是", "否", "是"]

for i, m in enumerate(models):
    table[f"({i+1})"].extend([
        city_fe[i],
        year_fe[i],
        int(m.nobs),
        f"{m.rsquared_adj:.4f}"
    ])

result = pd.DataFrame(table)

print("\n===== DID基准回归结果 =====\n")
print(result)
