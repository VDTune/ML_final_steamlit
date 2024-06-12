import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Thiết lập tiêu đề chính
st.title("Ứng dụng Machine Learning")

def display_statistics(df, target_variable, independent_variables):
    st.write("## Thống kê của biến mục tiêu")
    st.write(df[target_variable].value_counts())

    st.write("## Thống kê của các biến độc lập")
    for col in independent_variables:
        st.write(df[col].describe())

def display_correlation(df):
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    data = df[numeric_columns].copy()
    correlation_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Biểu đồ tương quan giữa các thuộc tính với biến lớp", fontsize=15, fontweight="bold")
    st.pyplot(plt.gcf())

def preprocess_data(df, target_variable, independent_variables):
    df = df.dropna().reset_index(drop=True)
    X = df[independent_variables].copy()

    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = le.fit_transform(X[col])
    
    Y = df[target_variable].copy()
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

from sklearn.neighbors import NearestNeighbors

def recommend(user_id, df, user_col, song_col, rating_col, n_recommendations, df_movies):
    # Tạo ma trận tiện ích
    utility_matrix = df.pivot_table(index=user_col, columns=song_col, values=rating_col)
    
    # Xử lý các giá trị NaN bằng median của từng cột
    imputer = SimpleImputer(strategy='median')
    utility_matrix_imputed = pd.DataFrame(imputer.fit_transform(utility_matrix), index=utility_matrix.index, columns=utility_matrix.columns)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    utility_matrix_scaled = pd.DataFrame(scaler.fit_transform(utility_matrix_imputed), index=utility_matrix_imputed.index, columns=utility_matrix_imputed.columns)
    
    # Tạo mô hình KNN
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(utility_matrix_scaled)

    user_id = int(user_id)
    st.write(f"User ID entered: {user_id}, type: {type(user_id)}")
    # Kiểm tra xem user_id có trong ma trận tiện ích không
    if user_id not in utility_matrix_scaled.index:
        raise ValueError("User ID not found in the dataset")
    
    # Tìm kiếm hàng xóm gần nhất
    user_index = utility_matrix_scaled.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors([utility_matrix_scaled.iloc[user_index, :]], n_neighbors=n_recommendations + 1)
    
    # Tạo danh sách các bài hát gợi ý
    recommendations = []
    for i in range(1, len(distances.flatten())):
        song_id = utility_matrix_scaled.columns[indices.flatten()[i]]
        # Tìm tiêu đề phim tương ứng
        title = df_movies.loc[df_movies[song_col] == song_id, 'title'].values[0]
        recommendations.append(title)
    
    return recommendations

def keyword_recommendation(df, keyword_col, num_recommendations):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df[keyword_col].astype(str))

    # Tính toán độ tương đồng cosine
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Lấy các mục có độ tương đồng cao nhất
    top_indices = cosine_sim[-1].argsort()[:-num_recommendations-1:-1]
    recommendations = df.iloc[top_indices][keyword_col].tolist()

    return recommendations

# Tải file lên từ thanh bên
uploaded_file = st.sidebar.file_uploader("Chọn CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.where(pd.notnull(df), None)

    model_type = st.sidebar.selectbox("Chọn loại mô hình", [
        "Logistic Regression", "KNN", "Random Forest", "Decision Tree", "Linear Regression"])

    if model_type == "KNN":
        k = st.sidebar.number_input("Chọn K", 1, len(df), 5)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_type == "Linear Regression":
        model = LinearRegression()

    target_variable = st.sidebar.selectbox("Chọn biến mục tiêu", df.columns, index=len(df.columns) - 1)
    independent_variables = st.sidebar.multiselect("Chọn biến độc lập", df.columns, default=list(df.columns.drop(target_variable)))

    # Add tabs
    tabs = st.tabs(["Thông tin dataset", "Huấn luyện mô hình", "Dự đoán", "Đề xuất theo đánh giá", "Gợi ý từ khóa"])

    with tabs[0]:
        st.write("## Thông tin Dataset")
        st.write(df)
        st.write(f"Số dòng: {df.shape[0]}")
        st.write(f"Số cột: {df.shape[1]}")
        st.write(f"Các cột: {df.columns.tolist()}")
        display_statistics(df, target_variable, independent_variables)
    with tabs[1]:
        if st.button("Huấn luyện mô hình"):
            if not independent_variables:
                st.error("Vui lòng chọn ít nhất một biến độc lập.")
            else:
                try:
                    X_train, X_test, Y_train, Y_test = preprocess_data(df, target_variable, independent_variables)
                    X_train, X_test = scale_data(X_train, X_test)

                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_test)

                    st.write("## Huấn luyện mô hình")
                    if model_type in ["Logistic Regression", "KNN", "Random Forest", "Decision Tree"]:
                        accuracy = accuracy_score(Y_test, Y_pred)
                        precision = precision_score(Y_test, Y_pred, average="weighted")
                        recall = recall_score(Y_test, Y_pred, average="weighted")
                        f1 = f1_score(Y_test, Y_pred, average="weighted")

                        st.write("## Đánh giá Mô hình")
                        st.write(f"Độ chính xác: {accuracy:.2f}")
                        st.write(f"Độ chính xác (Precision): {precision:.2f}")
                        st.write(f"Khả năng hồi đáp (Recall): {recall:.2f}")
                        st.write(f"F1 Score: {f1:.2f}")

                        st.write("## Confusion matrix")
                        conf_matrix = confusion_matrix(Y_test, Y_pred)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g")
                        plt.xlabel("Nhãn dự đoán")
                        plt.ylabel("Nhãn thực")
                        plt.title("Ma trận nhầm lẫn")
                        st.pyplot(plt.gcf())

                        st.write("## Biểu đồ tương quan giữa các thuộc tính với biến lớp")
                        display_correlation(df)

                        st.write("## Biểu đồ scatter plot của các cặp biến độc lập")
                        scatter_plot = sns.pairplot(df[independent_variables])
                        st.pyplot(scatter_plot)

                        st.write("## Biểu đồ phân phối của các biến độc lập")
                        for col in independent_variables:
                            plt.figure(figsize=(4, 2))
                            sns.histplot(df[col], kde=True)
                            plt.xlabel(col)
                            plt.ylabel("Số lượng")
                            plt.title(f"Phân phối của {col}")
                            st.pyplot(plt.gcf())

                    elif model_type == "Linear Regression":
                        mae = mean_absolute_error(Y_test, Y_pred)
                        mse = mean_squared_error(Y_test, Y_pred)
                        r2 = r2_score(Y_test, Y_pred)

                        st.write("## Đánh giá Mô hình")
                        st.write(f"Mean Absolute Error: {mae:.2f}")
                        st.write(f"Mean Squared Error: {mse:.2f}")
                        st.write(f"R² Score: {r2:.2f}")

                        if len(independent_variables) == 1:
                            plt.figure(figsize=(10, 6))
                            plt.scatter(X_test[:, 0], Y_test, color="black", label="Data")
                            plt.plot(X_test[:, 0], Y_pred, color="blue", linewidth=3, label="Regression line")
                            plt.xlabel(independent_variables[0])
                            plt.ylabel(target_variable)
                            plt.title("Linear Regression")
                            plt.legend()
                            st.pyplot(plt)
                        elif len(independent_variables) == 2:
                            fig = plt.figure(figsize=(10, 6))
                            ax = fig.add_subplot(111, projection="3d")
                            ax.scatter(X_test[:, 0], X_test[:, 1], Y_test, color="black", label="Data")
                            ax.set_xlabel(independent_variables[0])
                            ax.set_ylabel(independent_variables[1])
                            ax.set_zlabel(target_variable)
                            ax.set_title("Multiple Linear Regression")
                            ax.view_init(45, 0)
                            ax.legend()
                            st.pyplot(fig)
                        else:
                            st.write("Không thể vẽ đồ thị với hơn 2 biến độc lập.")
                    # Store the trained model for prediction use
                    st.session_state['model'] = model
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")
    with tabs[2]:
        st.write("## Dự đoán")
        input_data = {}
        for col in independent_variables:
            valueInf = df.iloc[1][col]
            value = st.text_input(f"nhập dữ liệu cho '{col}':", valueInf)
            input_data[col] = value

        if st.button("Dự đoán"):
            if 'model' not in st.session_state:
                st.error("Mô hình chưa được khởi tạo. Hãy khởi tạo mô hình trước.")
            else:
                try:
                    model = st.session_state['model']
                    input_df = pd.DataFrame([input_data])

                    if input_df.isnull().values.any():
                        st.error("Vui lòng điền vào tất cả các trường.")

                    le = LabelEncoder()
                    for col in input_df.columns:
                        if input_df[col].dtype == "object":
                            le.fit(input_df[col])
                            input_df[col] = le.transform(input_df[col])

                    prediction = model.predict(input_df)
                    st.write("## Kết quả dự đoán")
                    st.write(f"{target_variable}: {prediction[0]}")
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
    
    with tabs[3]:
        # item_index = st.sidebar.number_input("Chọn chỉ số mục tiêu (index)", min_value=0, max_value=len(df)-1, value=0)
        num_recommendations = st.slider("Số lượng gợi ý", 1, 10, 5)

        # Hiển thị danh sách các cột để người dùng chọn
        user_col = st.selectbox("Select User Column", df.columns)
        song_col = st.selectbox("Select Song Column", df.columns)
        rating_col = st.selectbox("Select Rating Column", df.columns)
        # Nhập user_id từ người dùng
        user_id = st.text_input("Enter User ID")
        
        if st.button("Gợi ý theo đánh giá"):
            try:
                movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
                recommendations = recommend(user_id, df, user_col, song_col, rating_col, num_recommendations, movies)
                st.write("## Kết quả gợi ý")
                st.write(recommendations)
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi gợi ý: {e}")
    with tabs[4]:      
        keyword_col = st.selectbox("Chọn cột từ khóa", df.columns, key="keyword_column_selectbox")
        num_recommendations = st.slider("Số lượng gợi ý", 1, 10, 5, key="num_recommendations_slider")

        if st.button("Gợi ý theo từ khóa", key="keyword_recommendation_button"):
            try:
                recommendations = keyword_recommendation(df, keyword_col, num_recommendations)
                st.write(f"Kết quả gợi ý theo từ khóa '{keyword_col}':")
                st.write(recommendations)
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi gợi ý: {e}")
    
    
    # if st.sidebar.button("Gợi ý theo từ khóa"):