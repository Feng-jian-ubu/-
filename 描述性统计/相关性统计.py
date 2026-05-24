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


corr = df[all_vars].corr().round(4)

corr_copy = corr.copy()

for i in range(len(corr_copy.columns)):
    for j in range(i+1, len(corr_copy.columns)):
        corr_copy.iloc[i, j] = np.nan

for i in range(len(corr_copy.columns)):
    corr_copy.iloc[i, i] = 1.0

print("\n===== 相关性分析 =====\n")
print(corr_copy)
