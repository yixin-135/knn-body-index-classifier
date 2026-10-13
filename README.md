# K 近邻（KNN）身体指数分类 · KNN Body-Index Classifier

> 使用 K 近邻算法，根据**性别 / 身高 / 体重**预测**身体指数（Index）**，是一个多分类任务。

## 任务说明

- 数据集列：`Gender`（Male/Female）、`Height`、`Weight`、`Index`（目标类别）
- 目标：用前三个特征预测 `Index`

## 处理流程

```
CSV 加载 → LabelEncoder 编码 Gender → train_test_split（stratify 分层划分）
→ StandardScaler 标准化 → KNN 训练 → 预测 → 评估
```

## 关键点

| 步骤 | 做法 | 原因 |
|------|------|------|
| 类别编码 | `LabelEncoder` 把 Male/Female 变成 0/1 | KNN 靠距离计算，字符串无法参与 |
| 划分数据 | `train_test_split(..., stratify=y)` | 分层抽样，保证各类别比例一致 |
| 特征标准化 | 训练集 `fit_transform`，测试集只用 `transform` | 身高（~170）与体重（~70）量纲差异大，必须标准化 |
| 模型参数 | `n_neighbors=7, metric="euclidean", weights="distance", p=2` | 奇数 K 避免平局，距离加权降低远点干扰 |
| 评估指标 | Accuracy / Precision / Recall / F1（weighted） | 多分类综合评估 |

## 运行环境

```bash
pip install pandas scikit-learn
```

## 数据说明（重要）

- 代码读取的是 `500_Person_Gender_Height_Weight_Index.csv`（Kaggle 公开数据集）。
- 该 CSV **未随仓库上传**（版权与体积原因），请自行到 Kaggle 搜索
  「500 Person Gender-Height-Weight-Body Mass Index」下载，并放到与本脚本同一目录。

## 运行方法

```bash
python main.py
```

> 注意：`main.py` 中定义的是 `run_knn_person_classification(train_size, random_state)` 函数，
> 需要自行调用（例如 `run_knn_person_classification(0.8, 42)`），脚本本身没有 `__main__` 入口。

## 目录结构

```
.
├── main.py                                     # KNN 分类主程序
├── 500_Person_Gender_Height_Weight_Index.csv   # 数据集（需自行下载，未上传）
└── README.md
```

## 说明

- 本项目为《人工智能编程实践》课程实训作业。
