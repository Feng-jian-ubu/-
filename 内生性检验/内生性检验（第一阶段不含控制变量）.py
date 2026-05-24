import pandas as pd
import statsmodels.api as sm


did_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\DID.xlsx"
road_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\道路总面积.xlsx"

did = pd.read_excel(did_path)
road = pd.read_excel(road_path)

did.columns = did.columns.str.strip()
road.columns = road.columns.str.strip()


for df_temp in [did, road]:
    df_temp["城市"] = df_temp["城市"].astype(str).str.strip()
    df_temp["年份"] = df_temp["年份"].astype(int)

df = did.merge(road, on=["城市", "年份"], how="inner")

print("合并后的列名：")
print(df.columns)

did_col = "DID"
road_col = "道路面积"


df["Time_IV"] = (df["年份"] >= 2020).astype(int)
df["Road_Time"] = df[road_col] * df["Time_IV"]

df = df.dropna(subset=[did_col, "Road_Time", "城市", "年份"])


city_dummies = pd.get_dummies(df["城市"], prefix="city", drop_first=True)
year_dummies = pd.get_dummies(df["年份"], prefix="year", drop_first=True)

X = pd.concat(
    [
        df[["Road_Time"]],
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


print("\n========== 第一阶段回归结果 ==========")

print(f"常数项: {result.params['const']:.4f}{add_stars(result.pvalues['const'])}")
print(f"标准误: ({result.bse['const']:.4f})")
print(f"t值: {result.tvalues['const']:.4f}")
print(f"p值: {result.pvalues['const']:.4f}")

print("--------------------------------------")

print(f"Road_Time: {result.params['Road_Time']:.4f}{add_stars(result.pvalues['Road_Time'])}")
print(f"标准误: ({result.bse['Road_Time']:.4f})")
print(f"t值: {result.tvalues['Road_Time']:.4f}")
print(f"p值: {result.pvalues['Road_Time']:.4f}")

print("--------------------------------------")

print(f"第一阶段F值: {f_value:.4f}{add_stars(f_pvalue)}")
print(f"F检验p值: {f_pvalue:.4f}")
