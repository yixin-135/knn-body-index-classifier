#  请在此处编写代码，实现使用 KNN 算法分类
# ====================Begin===================
# 导入必要的库
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_knn_person_classification(train_size,random_state):

    # ==================== 1. 数据加载 ====================
    df = pd.read_csv(
        "500_Person_Gender_Height_Weight_Index.csv",
    )

    # ==================== 2. 生成自变量和因变量 ====================
    X = df[["Gender","Height","Weight"]]

    y = df["Index"]

    # ==================== 3. 将 Gender 编码为数值 ====================
    # 假设 df 是你的数据集，feature_name 是待编码的类别特征列名
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])
    X = df[["Gender","Height","Weight"]]

    # ==================== 4. 数据集划分 ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        stratify=y  # 分层划分保证每个 Index 类别比例相同
    )

    # ==================== 5. 特征标准化 ====================
    # 初始化标准化器
    scaler = StandardScaler()
    # 对特征矩阵 X1和X2 进行标准化
    X_train_scaled = scaler.fit_transform(X_train)  # 在训练集 X1 上拟合（fit）标准化参数，并对训练集特征进行转换。
    X_test_scaled = scaler.transform(X_test)      # 使用在训练集上拟合得到的参数，对测试集 X2 进行相同的标准化转换。

    # ==================== 6. KNN 模型训练 ====================
    # 初始化模型
    model = KNeighborsClassifier(
        n_neighbors=7,
        metric="euclidean",
        weights="distance",
        p=2,
        algorithm="auto"
    )

    # 模型训练
    model.fit(X_train_scaled, y_train)

    # 模型预测
    y_pred = model.predict(X_test)

    # ==================== 7. 模型预测与评估 ====================

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    return float(accuracy), float(precision), float(recall), float(f1)
# ====================End===================