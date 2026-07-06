# CVD-Risk-Prediction
Cardiovascular Disease Risk Prediction Based on Logistic Regression & Random Forest
基于机器学习的心血管疾病10年患病风险预测（机器学习课程设计）

## 项目简介
本项目使用Framingham心脏研究医疗数据集，搭建二分类预测模型，实现冠心病早期辅助筛查。完整复现机器学习建模全流程：缺失/异常值清洗、EDA可视化、特征标准化、分层数据集划分、5折交叉验证、GridSearch网格搜索超参数优化，对比逻辑回归、随机森林模型性能，输出混淆矩阵、ROC曲线、特征重要性图表。
项目可拓展：模型量化后基于MindSpore Lite部署至鸿蒙智能穿戴设备，实现端侧离线健康风险评估。

## 项目目录
- code/ 核心训练代码
- data/ 数据集文件
- photo_data/ 实验生成全部可视化图表
- 基于逻辑回归与随机森林的心血管疾病风险预测研究.docx 完整实验报告

## 环境依赖
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

## 运行说明
1. VSCode打开项目根目录
2. 将数据集放入data文件夹
3. 运行code/heart_pred.py，自动输出5张分析图表至pics文件夹

## 实验内容
1. 数据预处理：均值/众数填充缺失值、医学阈值过滤异常样本
2. EDA：相关性热力图、分组箱线图分析生理指标与患病关联
3. 模型1：逻辑回归，5折交叉验证评估基线性能
4. 模型2：随机森林+网格搜索自动调参，输出特征重要性
5. 评估指标：Acc、Precision、Recall、F1、AUC、混淆矩阵、ROC曲线

## 拓展方向（适配鸿蒙端侧AI）
1. 使用skl2onnx导出训练模型，MindSpore Lite量化压缩
2. DevEco Studio开发ArkTS健康App，接入穿戴设备心率/血压数据
3. 鸿蒙分布式软总线实现多端健康报告同步
