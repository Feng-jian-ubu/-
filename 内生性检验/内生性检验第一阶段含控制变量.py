import pandas as pd
import statsmodels.api as sm


did_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\DID.xlsx"
road_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\道路总面积.xlsx"
control_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\控制变量.xlsx"


did = pd.read_excel(did_path)
road = pd.read_excel(road_path)
control = pd.read_excel(control_path)

did.columns = did.columns.str.strip()
road.columns = road.columns.str.strip()
control.columns = control.columns.str.strip()


for df_temp in [did, road, control]:
    df_temp["城市"] = df_temp["城市"].astype(str).str.strip()
    df_temp["年份"] = df_temp["年份"].astype(int)



df = did.merge(road, on=["城市", "年份"], how="inner")
df = df.merge(control, on=["城市", "年份"], how="inner")

print("合并后的列名：")
print(df.columns)


did_col = "DID"
road_col = "道路面积"

control_cols = [
    "人口规模",
    "经济发展水平",
    "对外开放水平",
    "城镇化率",
    "医疗卫生水平"
]


df["Time_IV"] = (df["年份"] >= 2020).astype(int)

df["Road_Time"] = df[road_col] * df["Time_IV"]


needed_cols = [did_col, "Road_Time", "城市", "年份"] + control_cols
df = df.dropna(subset=needed_cols)

city_dummies = pd.get_dummies(df["城市"], prefix="city", drop_first=True)
year_dummies = pd.get_dummies(df["年份"], prefix="year", drop_first=True)

X = pd.concat(
    [
        df[["Road_Time"] + control_cols],
        city_dummies,
        year_dummies
    ],
    axis=1
)

X = sm.add_constant(X)

X = X.astype(float)
y = df[did_col].astype(float)


model = sm.OLS(y, X)
result = model.fit()


def add_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return ""

f_test = result.f_test("Road_Time = 0")

f_value = float(f_test.fvalue)
f_pvalue = float(f_test.pvalue)


print("\n========== 第一阶段回归结果：加入控制变量 ==========")

vars_to_print = ["const", "Road_Time"] 

name_map = {
    "const": "常数项",
    "Road_Time": "Road_Time",
    "人口规模": "人口规模",
    "经济发展水平": "经济发展水平",
    "对外开放水平": "对外开放水平",
    "城镇化率": "城镇化率",
    "医疗卫生水平": "医疗卫生水平"
}

for var in vars_to_print:
    print(f"{name_map[var]}: {result.params[var]:.4f}{add_stars(result.pvalues[var])}")
    print(f"标准误: ({result.bse[var]:.4f})")
    print(f"t值: {result.tvalues[var]:.4f}")
    print(f"p值: {result.pvalues[var]:.4f}")
    print("--------------------------------------")

print(f"第一阶段F值: {f_value:.4f}{add_stars(f_pvalue)}")
print(f"F检验p值: {f_pvalue:.4f}")

print("城市固定效应: 是")
print("年份固定效应: 是")
print("控制变量: 是")
print(f"样本量: {int(result.nobs)}")
print(f"R²: {result.rsquared:.4f}")
