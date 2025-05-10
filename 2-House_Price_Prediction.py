# Command: "streamlit run 2-House_Price_Prediction.py" to run this file.

# Import necessary libraries
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import streamlit as st  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   # type: ignore

# Streamlit UI
st.title("PRODIGY INTERNSHIP PROGRAME")
st.title("🏡 House Price Prediction System")
st.subheader("Hey there 👋 You are Most Welcome 😊!!")
st.write("If you want to predict the price of your house, you're at the **right place!** ✅ Give it a try!")
st.subheader("PROBLEM STATEMENT:")
st.write("Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms")

# Load Dataset
st.warning("Must choose train.csv file as per the Problem statement. It is attached in Github")
uploaded_file = st.file_uploader("Upload a train.csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview (First 5 rows):", df.head(5))

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols:
        label_encoders = {}
        for col in categorical_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            label_encoders[col] = encoder
        st.write("✅ Categorical values encoded.")

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df.fillna(df.mean(), inplace=True)
        st.write("✅ Missing values handled by filling with column mean.")

    # Select target and feature columns
    st.warning("Must choose 'SalePrice' as Target column for this Problem statement otherwise Model may fluctuate")
    target_column = st.selectbox("🎯 Select Target Column", df.columns)
    st.warning("Must choose 1stFlrSF, 2ndFlrSF, GrLivArea, TotalBsmtSF, BsmtFinSF1, BedroomAbvGr, BsmtFullBath and FullBath in Feature column for this Problem statement otherwise Model may fluctuate because data processing is done base on these particular features.")
    features = st.multiselect("📊 Select Feature Columns", [col for col in df.columns if col != target_column])

    if features:
        df = df[[target_column] + features]
        st.write("Filtered Dataset Preview:", df.head())

        # Feature Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df.drop(columns=[target_column]))

        # Define X (features) and y (target)
        X = pd.DataFrame(scaled_features, columns=df.drop(columns=[target_column]).columns)
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression model
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)

        # Predictions on test data
        linear_y_preds = lin_model.predict(X_test)

        # Model Evaluation
        st.write(f"📉 Mean Absolute Error: {mean_absolute_error(y_test, linear_y_preds):.2f}")
        st.write(f"📊 Mean Squared Error: {mean_squared_error(y_test, linear_y_preds):.2f}")
        st.write(f"📈 R² Score: {r2_score(y_test, linear_y_preds):.2f}")

        # User input for prediction
        st.subheader("🔮 Predict House Price Based on User Input")

        input_data = {feature: st.text_input(f"Enter categorical {feature}") if feature in categorical_cols
                      else st.number_input(f"Enter numerical {feature}", value=0)
                      for feature in features}

        if st.button("Predict"):
            new_dataframe = pd.DataFrame([input_data])

            # Encode categorical features
            for col in categorical_cols:
                if col in new_dataframe:
                    new_dataframe[col] = label_encoders[col].transform([new_dataframe[col][0]])

            # Ensure consistency with training columns
            missing_cols = [col for col in X_train.columns if col not in new_dataframe.columns]

            for col in missing_cols:
                new_dataframe[col] = 0

            new_dataframe = new_dataframe[X_train.columns]

            # Standardize input features
            scaled_new_data = scaler.transform(new_dataframe)

            # Prediction
            prediction = lin_model.predict(scaled_new_data)
            st.success(f"💰 Estimated House Price: **{prediction[0]:.2f}**")