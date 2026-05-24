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

did_var = "DID"

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

print("\n===== 平行趋势检验 =====\n")
def star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return ""

df["treat_dummy"] = df["Treat_Time"]

policy_year = df[df["treat_dummy"] == 1].groupby("城市")["年份"].min()

df["policy_year"] = df["城市"].map(policy_year)


df["event_time"] = df["年份"] - df["policy_year"]


min_k = -5
max_k = 3

df_event = df.copy()


event_vars = []

for k in range(min_k, max_k + 1):

    if k == -1:
        continue

    if k < 0:
        var_name = f"event_m{abs(k)}"  
    else:
        var_name = f"event_{k}"       

    df_event[var_name] = ((df_event["event_time"] == k) * 1).fillna(0)

    event_vars.append(var_name)


formula_event = (
    f"{y} ~ " +
    " + ".join(event_vars + controls) +
    " + C(城市) + C(年份)"
)

event_model = smf.ols(formula_event, data=df_event).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_event["城市"]}
)


event_result = []

for k in range(min_k, max_k + 1):

    if k == -1:
        continue

    if k < 0:
        var = f"event_m{abs(k)}"
    else:
        var = f"event_{k}"

    coef = event_model.params.get(var, float("nan"))
    se = event_model.bse.get(var, float("nan"))
    p = event_model.pvalues.get(var, 1)

    event_result.append([
        k,
        f"{coef:.4f}{star(p)}",
        f"({se:.4f})"
    ])

event_df = pd.DataFrame(event_result, columns=["相对时间", "系数", "标准误"])

print(event_df)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.rcParams['axes.unicode_minus'] = False    

event_df["coef"] = event_df["系数"].str.replace(r"\*+", "", regex=True).astype(float)
event_df["se"] = event_df["标准误"].str.replace(r"[()]", "", regex=True).astype(float)

x = event_df["相对时间"]
y = event_df["coef"]
se = event_df["se"]

ci_upper = y + 1.96 * se
ci_lower = y - 1.96 * se


plt.figure()

plt.errorbar(x, y, yerr=1.96*se, fmt='o', capsize=4)

plt.plot(x, y)

plt.axhline(0)  
plt.axvline(-1)  
plt.xlabel("政策时点")
plt.ylabel("政策动态效应")
plt.title("平行趋势检验")

plt.show()
