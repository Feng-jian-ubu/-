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

df.rename(columns={"Treat×Time": "Treat_Time"}, inplace=True)


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
    "DID": "政策变量",
    "人口规模": "户籍人口（取对数）",
    "经济发展水平": "人均地区生产总值（取对数）",
    "对外开放水平": "实际利用外资额/地区生产总值",
    "城镇化率": "非农业人口/户籍人口",
    "医疗卫生水平": "每百人医院、卫生院床位"
}


desc = df[all_vars].describe().T

desc = desc[["mean", "std", "min", "max"]]

desc.columns = ["均值", "标准差", "最小值", "最大值"]

desc.insert(0, "变量含义", [var_label[i] for i in desc.index])

desc.insert(0, "变量", desc.index)

desc.reset_index(drop=True, inplace=True)

print("\n===== 描述性统计 =====\n")
print(desc)
