# 导入全部依赖库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve

# 解决VSCode绘图中文乱码（必加）
# 替换原来的两行字体代码
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 读取数据集 =====================
df = pd.read_csv("../data/framingham_heart_study.csv")
print("数据集整体尺寸：", df.shape)
print("\n各字段缺失值统计：")
print(df.isnull().sum())
print("\n目标变量 TenYearCHD 分布（0=无患病，1=10年内冠心病）：")
print(df["TenYearCHD"].value_counts(normalize=True))
print("\n数据描述性统计：")
print(df.describe())

# ===================== 2. 缺失值填充处理 =====================
num_cols = ["cigsPerDay", "totChol", "BMI", "heartRate", "glucose"]
cat_cols = ["education", "BPMeds"]
# 数值特征均值填充
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)
# 类别特征众数填充
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ===================== 3. 医学逻辑过滤异常值 =====================
df = df[(df["sysBP"] > 0) & (df["diaBP"] > 0)]
df = df[(df["BMI"] > 10) & (df["BMI"] < 60)]
df = df[(df["heartRate"] >= 30) & (df["heartRate"] <= 220)]
print(f"\n异常值清洗后剩余样本量：{df.shape[0]}")

# ===================== 4. EDA可视化分析 =====================
# 4.1 特征相关性热力图
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("特征相关性热力图")
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# 4.2 患病/健康分组箱线图（年龄、收缩压、BMI）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x="TenYearCHD", y="age", data=df, ax=axes[0])
axes[0].set_title("不同患病状态年龄分布")
sns.boxplot(x="TenYearCHD", y="sysBP", data=df, ax=axes[1])
axes[1].set_title("不同患病状态收缩压分布")
sns.boxplot(x="TenYearCHD", y="BMI", data=df, ax=axes[2])
axes[2].set_title("不同患病状态BMI分布")
plt.tight_layout()
plt.savefig("boxplot.png", dpi=300)
plt.show()

# ===================== 5. 特征工程 & 数据集划分 =====================
# 分离特征与标签
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]
# 数值标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 分层抽样划分训练集70%、测试集30%，保证患病比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ===================== 6. 模型1：逻辑回归 + 5折交叉验证 =====================
lr = LogisticRegression(max_iter=1000)
# 5折交叉验证评估AUC稳定性
cv_auc_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring="roc_auc")
print(f"\n逻辑回归5折交叉验证AUC均值：{cv_auc_lr.mean():.4f}")
# 训练并预测
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# ===================== 7. 模型2：随机森林 + GridSearch网格调参 =====================
rf = RandomForestClassifier(random_state=42)
# 超参数搜索空间
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train, y_train)
print("\n随机森林最优超参数组合：", grid_search.best_params_)
# 最优模型预测
rf_best = grid_search.best_estimator_
y_pred_rf = rf_best.predict(X_test)
y_proba_rf = rf_best.predict_proba(X_test)[:, 1]

# ===================== 8. 模型评估指标计算 =====================
def get_eval_metrics(y_true, y_pred, y_proba):
    acc = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred), 4)
    recall = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    auc = round(roc_auc_score(y_true, y_proba), 4)
    return [acc, precision, recall, f1, auc]

lr_result = get_eval_metrics(y_test, y_pred_lr, y_proba_lr)
rf_result = get_eval_metrics(y_test, y_pred_rf, y_proba_rf)

# 打印指标对比表格
print("\n========== 两种模型性能对比表 ==========")
metrics_name = ["准确率Acc", "精确率Precision", "召回率Recall", "F1分数", "AUC值"]
print(f"{'评估指标':<16}{'逻辑回归':<12}{'最优随机森林':<12}")
for idx, name in enumerate(metrics_name):
    print(f"{name:<16}{lr_result[idx]:<12}{rf_result[idx]:<12}")

# ===================== 9. 随机森林混淆矩阵 =====================
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("随机森林混淆矩阵")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ===================== 10. ROC曲线对比 =====================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"逻辑回归 AUC={lr_result[4]}")
plt.plot(fpr_rf, tpr_rf, label=f"随机森林 AUC={rf_result[4]}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("假阳性率 FPR")
plt.ylabel("真阳性率 TPR")
plt.title("两种模型ROC曲线对比")
plt.legend()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# ===================== 11. 特征重要性排序图 =====================
feature_names = X.columns
feature_importance = rf_best.feature_importances_
# 按重要性降序排序
sorted_index = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_names)), feature_importance[sorted_index])
plt.xticks(range(len(feature_names)), feature_names[sorted_index], rotation=45)
plt.title("心血管风险特征重要性排序")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()