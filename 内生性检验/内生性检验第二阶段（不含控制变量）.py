import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

eco_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\城市生态韧性\熵权法_城市生态韧性.xlsx"
did_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\DID.xlsx"
road_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\道路总面积.xlsx"


eco = pd.read_excel(eco_path)
did = pd.read_excel(did_path)
road = pd.read_excel(road_path)

eco.columns = eco.columns.str.strip()
did.columns = did.columns.str.strip()
road.columns = road.columns.str.strip()


for df_temp in [eco, did, road]:
    df_temp["城市"] = df_temp["城市"].astype(str).str.strip()
    df_temp["年份"] = df_temp["年份"].astype(int)


df = eco.merge(did, on=["城市", "年份"], how="inner")
df = df.merge(road, on=["城市", "年份"], how="inner")

print("合并后的列名：")
print(df.columns)


y_col = "Eco_Resilience"
did_col = "DID"
road_col = "道路面积"


df["Time_IV"] = (df["年份"] >= 2020).astype(int)

df["Road_Time"] = df[road_col] * df["Time_IV"]



df = df.dropna(subset=[y_col, did_col, "Road_Time", "城市", "年份"])

city_dummies = pd.get_dummies(df["城市"], prefix="city", drop_first=True)
year_dummies = pd.get_dummies(df["年份"], prefix="year", drop_first=True)

exog = pd.concat(
    [
        pd.Series(1, index=df.index, name="const"),
        city_dummies,
        year_dummies
    ],
    axis=1
)

exog = exog.astype(float)


y = df[y_col].astype(float)

endog = df[[did_col]].astype(float)

instr = df[["Road_Time"]].astype(float)



iv_model = IV2SLS(
    dependent=y,
    exog=exog,
    endog=endog,
    instruments=instr
)

iv_result = iv_model.fit(cov_type="robust")


print("\n========== 第二阶段工具变量回归结果：不含控制变量 ==========")

vars_to_print = ["const", did_col]

name_map = {
    "const": "常数项",
    did_col: "DID"
}

for var in vars_to_print:
    print(f"{name_map[var]}: {iv_result.params[var]:.4f}{add_stars(iv_result.pvalues[var])}")
    print(f"标准误: ({iv_result.std_errors[var]:.4f})")
    print(f"t值: {iv_result.tstats[var]:.4f}")
    print(f"p值: {iv_result.pvalues[var]:.4f}")
    print("--------------------------------------")

print("城市固定效应: 是")
print("年份固定效应: 是")
print("控制变量: 否")
print(f"样本量: {int(iv_result.nobs)}")
