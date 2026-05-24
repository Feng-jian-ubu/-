import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import warnings

warnings.filterwarnings("ignore")

eco_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\城市生态韧性\熵权法_城市生态韧性.xlsx"
did_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\DID.xlsx"
control_path = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\TJJM20260418190871\数据及其他-TJJM20260418190871\控制变量.xlsx"

output_dir = r"D:\苦命大学生的portrait\课程之外\统计建模大赛\_本地初稿_统计建模大赛论文"
os.makedirs(output_dir, exist_ok=True)

eco_df = pd.read_excel(eco_path)
did_df = pd.read_excel(did_path)
control_df = pd.read_excel(control_path)

eco_df = eco_df.rename(columns={
    eco_df.columns[0]: '城市',
    eco_df.columns[1]: '年份'
})

did_df = did_df.rename(columns={
    did_df.columns[0]: '城市',
    did_df.columns[1]: '年份'
})

control_df = control_df.rename(columns={
    control_df.columns[0]: '城市',
    control_df.columns[1]: '年份'
})

eco_df.columns = eco_df.columns.astype(str).str.strip()
did_df.columns = did_df.columns.astype(str).str.strip()
control_df.columns = control_df.columns.astype(str).str.strip()

eco_df['城市'] = eco_df['城市'].astype(str).str.strip()
did_df['城市'] = did_df['城市'].astype(str).str.strip()
control_df['城市'] = control_df['城市'].astype(str).str.strip()

eco_df['年份'] = pd.to_numeric(eco_df['年份'], errors='coerce')
did_df['年份'] = pd.to_numeric(did_df['年份'], errors='coerce')
control_df['年份'] = pd.to_numeric(control_df['年份'], errors='coerce')

eco_df = eco_df.dropna(subset=['年份']).copy()
did_df = did_df.dropna(subset=['年份']).copy()
control_df = control_df.dropna(subset=['年份']).copy()

eco_df['年份'] = eco_df['年份'].astype(int)
did_df['年份'] = did_df['年份'].astype(int)
control_df['年份'] = control_df['年份'].astype(int)

print("生态韧性表列名：", eco_df.columns.tolist())
print("DID表列名：", did_df.columns.tolist())
print("控制变量表列名：", control_df.columns.tolist())

eco_var = 'Eco_Resilience'
did_var = 'DID'

control_vars = ['人口规模', '经济发展水平', '对外开放水平', '城镇化率', '医疗卫生水平']

if eco_var not in eco_df.columns:
    raise ValueError(f"生态韧性表中未找到列名：{eco_var}，请根据打印出的列名修改 eco_var。")

if did_var not in did_df.columns:
    raise ValueError(f"DID表中未找到列名：{did_var}，请根据打印出的列名修改 did_var。")

for var in control_vars:
    if var not in control_df.columns:
        raise ValueError(f"控制变量表中未找到列名：{var}")

df = pd.merge(
    eco_df[['城市', '年份', eco_var]],
    did_df[['城市', '年份', did_var]],
    on=['城市', '年份'],
    how='inner'
)

df = pd.merge(
    df,
    control_df[['城市', '年份'] + control_vars],
    on=['城市', '年份'],
    how='inner'
)

for col in [eco_var, did_var] + control_vars:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=[eco_var, did_var] + control_vars).copy()

print("\n合并后数据预览：")
print(df.head())
print("\n合并后样本量：", len(df))
print("城市数：", df['城市'].nunique())
print("年份范围：", df['年份'].min(), "-", df['年份'].max())

policy_year = 2019
df['post'] = (df['年份'] >= policy_year).astype(int)

treated_cities = df.loc[df[did_var] == 1, '城市'].unique()
n_treated = len(treated_cities)
all_cities = df['城市'].unique()

print("\n真实处理组城市数量：", n_treated)
print("总城市数量：", len(all_cities))

if n_treated == 0:
    raise ValueError("没有识别到真实处理组城市，请检查 DID 列。")

formula_real = (
    f"{eco_var} ~ {did_var} + "
    + " + ".join([f"Q('{v}')" for v in control_vars])
    + " + C(城市) + C(年份)"
)

model_real = smf.ols(formula=formula_real, data=df).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['城市']}
)

real_coef = model_real.params[did_var]
real_pval = model_real.pvalues[did_var]

print("\n========== 基准回归结果 ==========")
print("真实 DID 系数：", real_coef)
print("真实 DID p值：", real_pval)

n_rep = 500
coef_list = []
pval_list = []

np.random.seed(12345)

for i in range(n_rep):
    placebo_cities = np.random.choice(all_cities, size=n_treated, replace=False)

    df['placebo_treat'] = df['城市'].isin(placebo_cities).astype(int)
    df['placebo_DID'] = df['placebo_treat'] * df['post']

    formula_placebo = (
        f"{eco_var} ~ placebo_DID + "
        + " + ".join([f"Q('{v}')" for v in control_vars])
        + " + C(城市) + C(年份)"
    )

    try:
        model_placebo = smf.ols(formula=formula_placebo, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df['城市']}
        )

        coef_list.append(model_placebo.params['placebo_DID'])
        pval_list.append(model_placebo.pvalues['placebo_DID'])

    except Exception as e:
        print(f"第 {i+1} 次回归失败：{e}")
        coef_list.append(np.nan)
        pval_list.append(np.nan)

result_df = pd.DataFrame({
    'placebo_coef': coef_list,
    'placebo_pvalue': pval_list
})

result_df = result_df.dropna().copy()

print("\n成功完成的安慰剂回归次数：", len(result_df))

result_path = os.path.join(output_dir, "安慰剂检验结果.xlsx")
result_df.to_excel(result_path, index=False)

print("安慰剂检验明细已保存：", result_path)

summary_df = pd.DataFrame({
    '真实DID系数': [real_coef],
    '真实DID_p值': [real_pval],
    '安慰剂回归次数': [len(result_df)],
    '安慰剂系数均值': [result_df['placebo_coef'].mean()],
    '安慰剂系数标准差': [result_df['placebo_coef'].std()],
    '安慰剂p值均值': [result_df['placebo_pvalue'].mean()],
    'p值>0.1比例': [(result_df['placebo_pvalue'] > 0.1).mean()]
})

summary_path = os.path.join(output_dir, "安慰剂检验汇总结果.xlsx")
summary_df.to_excel(summary_path, index=False)

print("安慰剂检验汇总已保存：", summary_path)
print("\n========== 安慰剂检验汇总 ==========")
print(summary_df)

coef = result_df['placebo_coef'].values
pval = result_df['placebo_pvalue'].values

kde = gaussian_kde(coef)
x_vals = np.linspace(coef.min(), coef.max(), 300)
density = kde(x_vals)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.scatter(coef, pval, alpha=0.5)
ax1.set_xlabel('Placebo DID Coefficient', fontsize=12)
ax1.set_ylabel('P-value', fontsize=12)
ax1.axhline(y=0.1, linestyle='--', linewidth=1.2, label='p=0.1')
ax1.axvline(x=0, linestyle='--', linewidth=1.2, label='x=0')
ax1.axvline(x=real_coef, linestyle='--', linewidth=1.5, label=f'Real coef={real_coef:.4f}')
ax1.set_ylim(-0.02, 1.05)

ax2 = ax1.twinx()
ax2.plot(x_vals, density, linewidth=2)
ax2.set_ylabel('Kernel Density', fontsize=12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)

plt.title('Placebo Test', fontsize=14)
plt.tight_layout()

plot_path = os.path.join(output_dir, "安慰剂检验双坐标图.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print("双坐标图已保存：", plot_path)

mean_coef = result_df['placebo_coef'].mean()
p_over_01 = (result_df['placebo_pvalue'] > 0.1).mean()

print("\n========== 简要结论 ==========")
print(f"安慰剂系数均值：{mean_coef:.6f}")
print(f"p值大于0.1的比例：{p_over_01:.2%}")

if abs(mean_coef) < 0.01 and p_over_01 > 0.7:
    print("初步判断：安慰剂检验结果较理想，基准回归结论具有一定稳健性。")
else:
    print("初步判断：安慰剂检验结果一般，建议进一步检查模型设定、样本或变量构造。")
