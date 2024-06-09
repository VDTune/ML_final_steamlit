import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Thiết lập tiêu đề chính
st.title("Ứng dụng Machine Learning")


def display_statistics(df, target_variable, independent_variables):
    st.write("## Thống kê của biến mục tiêu")
    st.write(df[target_variable].value_counts())

    st.write("## Thống kê của các biến độc lập")
    for col in independent_variables:
        st.write(df[col].describe())


# Hàm tính toán và hiển thị biểu đồ tương quan giữa các thuộc tính và biến lớp
def display_correlation(df, target_variable, independent_variables):
    # Lọc ra các cột chứa dữ liệu numeric
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

    # Lọc DataFrame chỉ chứa các cột numeric và biến lớp
    data = df[[target_variable] + independent_variables].copy()
    data = data[numeric_columns]

    # Tính toán ma trận tương quan
    correlation_matrix = data.corr()

    # Vẽ biểu đồ heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(
        "biểu đồ tương quan giữa các thuộc tính với biến lớp",
        fontsize=15,
        fontweight="bold",
    )
    st.pyplot(plt.gcf())


# Tải file lên từ thanh bên
uploaded_file = st.sidebar.file_uploader("Chọn CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.where(pd.notnull(df), None)  # Thay thế NaN bằng None

    # Hiển thị thông tin dataset
    st.write("## Thông tin Dataset")
    st.write(df.head())
    st.write(f"Số dòng: {df.shape[0]}")
    st.write(f"Số cột: {df.shape[1]}")
    st.write(f"Các cột: {df.columns.tolist()}")

    # Chọn loại mô hình
    model_type = st.sidebar.selectbox(
        "Chọn loại mô hình",
        ["Logistic Regression", "KNN", "Random Forest", "Decision Tree"],
    )

    # Chọn biến mục tiêu
    target_variable = st.sidebar.selectbox(
        "Chọn biến mục tiêu", df.columns, index=len(df.columns) - 1
    )
    # Chọn biến độc lập
    independent_variables = st.sidebar.multiselect(
        "Chọn biến độc lập",
        df.columns,
        default=list(df.columns.drop(target_variable)),
    )

    if model_type == "KNN":
        k = st.sidebar.number_input("Chọn K", 1, len(df), 5)

    # Nút để bắt đầu huấn luyện mô hình
    if st.sidebar.button("Huấn luyện mô hình"):
        if not independent_variables:
            st.error("Vui lòng chọn ít nhất một biến độc lập.")
        else:
            X = df[independent_variables].copy()
            Y = df[target_variable].copy()

            # Mã hóa các biến phân loại
            le = LabelEncoder()
            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = le.fit_transform(X[col])
            if Y.dtype == "object":
                Y = le.fit_transform(Y)

            # Điền giá trị NaN bằng giá trị trung bình của cột
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)

            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )
            # Scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Chọn mô hình
            if model_type == "KNN":
                model = KNeighborsClassifier(n_neighbors=k)
            elif model_type == "Logistic Regression":
                model = LogisticRegression()
            elif model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier()

            # Huấn luyện mô hình
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            # Tính toán các chỉ số đánh giá mô hình
            accuracy = accuracy_score(Y_test, Y_pred)
            precision = precision_score(Y_test, Y_pred, average="weighted")
            recall = recall_score(Y_test, Y_pred, average="weighted")
            f1 = f1_score(Y_test, Y_pred, average="weighted")

            # Hiển thị thông kê và biểu đồ
            # display_statistics(df, target_variable, independent_variables)

            # Hiển thị các chỉ số đánh giá mô hình
            st.write("## Đánh giá Mô hình")
            st.write(f"Độ chính xác: {accuracy:.2f}")
            st.write(f"Độ chính xác (Precision): {precision:.2f}")
            st.write(f"Khả năng hồi đáp (Recall): {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

            # Ma trận nhầm lẫn
            st.write("## Ma trận nhầm lẫn")
            conf_matrix = confusion_matrix(Y_test, Y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g")
            plt.xlabel("Nhãn dự đoán")
            plt.ylabel("Nhãn thực")
            plt.title("Ma trận nhầm lẫn")
            st.pyplot(plt.gcf())
            st.write("## biểu đồ tương quan giữa các thuộc tính với biến lớp")
            display_correlation(df, target_variable, independent_variables)

            # Hiển thị biểu đồ scatter plot của các cặp biến độc lập
            st.write("## Biểu đồ scatter plot của các cặp biến độc lập")
            scatter_plot = sns.pairplot(df[independent_variables])
            st.pyplot(scatter_plot)
            # Hiển thị biểu đồ phân phối của các biến độc lập
            st.write("## Biểu đồ phân phối của các biến độc lập")
            for col in independent_variables:
                plt.figure(figsize=(4, 2))
                sns.histplot(df[col], kde=True)
                plt.xlabel(col)
                plt.ylabel("Số lượng")
                plt.title(f"Phân phối của {col}")
                st.pyplot(plt.gcf())


# Hàm tính toán và hiển thị thông kê
