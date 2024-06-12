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
from sklearn.neighbors import NearestNeighbors

# Set main title
st.title("Machine Learning Application")

def display_statistics(df, target_variable, independent_variables):
    st.write("## Target Variable Statistics")
    st.write(df[target_variable].value_counts())

    st.write("## Independent Variables Statistics")
    for col in independent_variables:
        st.write(df[col].describe())

def display_correlation(df):
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    data = df[numeric_columns].copy()
    correlation_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix", fontsize=15, fontweight="bold")
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

def recommend(user_id, df, user_col, song_col, rating_col, n_recommendations, df_movies):
    utility_matrix = df.pivot_table(index=user_col, columns=song_col, values=rating_col)
    imputer = SimpleImputer(strategy='median')
    utility_matrix_imputed = pd.DataFrame(imputer.fit_transform(utility_matrix), index=utility_matrix.index, columns=utility_matrix.columns)
    scaler = StandardScaler()
    utility_matrix_scaled = pd.DataFrame(scaler.fit_transform(utility_matrix_imputed), index=utility_matrix_imputed.index, columns=utility_matrix_imputed.columns)
    
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(utility_matrix_scaled)

    user_id = int(user_id)
    st.write(f"User ID entered: {user_id}, type: {type(user_id)}")
    if user_id not in utility_matrix_scaled.index:
        raise ValueError("User ID not found in the dataset")
    
    user_index = utility_matrix_scaled.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors([utility_matrix_scaled.iloc[user_index, :]], n_neighbors=n_recommendations + 1)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        song_id = utility_matrix_scaled.columns[indices.flatten()[i]]
        title = df_movies.loc[df_movies[song_col] == song_id, 'title'].values[0]
        recommendations.append(title)
    
    return recommendations

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.where(pd.notnull(df), None)

    model_type = st.sidebar.selectbox("Choose Model Type", [
        "Logistic Regression", "KNN", "Random Forest", "Decision Tree", "Linear Regression"])

    if model_type == "KNN":
        k = st.sidebar.number_input("Choose K", 1, len(df), 5)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_type == "Linear Regression":
        model = LinearRegression()

    target_variable = st.sidebar.selectbox("Choose Target Variable", df.columns, index=len(df.columns) - 1)
    independent_variables = st.sidebar.multiselect("Choose Independent Variables", df.columns, default=list(df.columns.drop(target_variable)))

    # Add tabs
    tabs = st.tabs(["Dataset Information", "Model Training", "Predictions"])

    with tabs[0]:
        st.write("## Dataset Information")
        st.write(df)
        st.write(f"Number of Rows: {df.shape[0]}")
        st.write(f"Number of Columns: {df.shape[1]}")
        st.write(f"Columns: {df.columns.tolist()}")
        display_statistics(df, target_variable, independent_variables)
        display_correlation(df)

    with tabs[1]:
        if st.button("Train Model"):
            if not independent_variables:
                st.error("Please choose at least one independent variable.")
            else:
                try:
                    X_train, X_test, Y_train, Y_test = preprocess_data(df, target_variable, independent_variables)
                    X_train, X_test = scale_data(X_train, X_test)

                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_test)

                    st.write("## Model Evaluation")

                    if model_type in ["Logistic Regression", "KNN", "Random Forest", "Decision Tree"]:
                        accuracy = accuracy_score(Y_test, Y_pred)
                        precision = precision_score(Y_test, Y_pred, average="weighted")
                        recall = recall_score(Y_test, Y_pred, average="weighted")
                        f1 = f1_score(Y_test, Y_pred, average="weighted")

                        st.write(f"Accuracy: {accuracy:.2f}")
                        st.write(f"Precision: {precision:.2f}")
                        st.write(f"Recall: {recall:.2f}")
                        st.write(f"F1 Score: {f1:.2f}")

                        st.write("## Confusion Matrix")
                        conf_matrix = confusion_matrix(Y_test, Y_pred)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g")
                        plt.xlabel("Predicted Label")
                        plt.ylabel("True Label")
                        plt.title("Confusion Matrix")
                        st.pyplot(plt.gcf())

                    elif model_type == "Linear Regression":
                        mae = mean_absolute_error(Y_test, Y_pred)
                        mse = mean_squared_error(Y_test, Y_pred)
                        r2 = r2_score(Y_test, Y_pred)

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
                            st.write("Cannot plot more than 2 independent variables.")

                    # Store the trained model for prediction use
                    st.session_state['model'] = model
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    with tabs[2]:
        st.write("## Make Predictions")
        input_data = {}
        for col in independent_variables:
            valueInf = df.iloc[1][col]
            value = st.text_input(f"Enter value for '{col}':", valueInf)
            input_data[col] = value

        if st.button("Predict"):
            if 'model' not in st.session_state:
                st.error("Model is not trained yet. Please train the model first.")
            else:
                try:
                    model = st.session_state['model']
                    input_df = pd.DataFrame([input_data])

                    if input_df.isnull().values.any():
                        st.error("Please fill in all the fields.")

                    le = LabelEncoder()
                    for col in input_df.columns:
                        if input_df[col].dtype == "object":
                            le.fit(input_df[col])
                            input_df[col] = le.transform(input_df[col])

                    prediction = model.predict(input_df)
                    st.write("## Prediction Result")
                    st.write(f"{target_variable}: {prediction[0]}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

    st.sidebar.subheader("Recommendations")
    num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
    user_col = st.sidebar.selectbox("Select User Column", df.columns)
    song_col = st.sidebar.selectbox("Select Song Column", df.columns)
    rating_col = st.sidebar.selectbox("Select Rating Column", df.columns)
    user_id = st.sidebar.text_input("Enter User ID")
    
    if st.sidebar.button("Get Recommendations"):
        try:
            movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
            recommendations = recommend(user_id, df, user_col, song_col, rating_col, num_recommendations, movies)
            st.write("## Recommendations")
            st.write(recommendations)
        except Exception as e:
            st.error(f"An error occurred during recommendation: {e}")
