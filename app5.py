import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00FFAA;
}

.stButton>button {
    background-color: #00FFAA;
    color: black;
    font-size: 18px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATASET ----------------

df = pd.read_csv("diabetes.csv")

# ---------------- SIDEBAR ----------------

st.sidebar.title("🩺 Navigation")

option = st.sidebar.radio(
    "Go to",
    ["Prediction", "Dataset", "Graphs", "About"]
)

# ---------------- DATA PREPARATION ----------------

X = df.drop(columns='Outcome', axis=1)

Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

# ---------------- MODELS ----------------

# Logistic Regression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, Y_train)

# Random Forest

rf_model = RandomForestClassifier()

rf_model.fit(X_train, Y_train)

# Decision Tree

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, Y_train)

# SVM

svm_model = SVC(probability=True)

svm_model.fit(X_train, Y_train)

# ---------------- ACCURACY ----------------

lr_accuracy = accuracy_score(
    lr_model.predict(X_test),
    Y_test
)

rf_accuracy = accuracy_score(
    rf_model.predict(X_test),
    Y_test
)

dt_accuracy = accuracy_score(
    dt_model.predict(X_test),
    Y_test
)

svm_accuracy = accuracy_score(
    svm_model.predict(X_test),
    Y_test
)

# ---------------- PREDICTION PAGE ----------------

if option == "Prediction":

    st.title("🩺 Diabetes Prediction System")

    st.subheader("Machine Learning Healthcare Dashboard")

    st.write("Enter patient health details below:")

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0
    )

    insulin = st.number_input(
        "Insulin Level",
        min_value=0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0
    )

    age = st.number_input(
        "Age",
        min_value=0
    )

    # ---------------- MODEL SELECTION ----------------

    model_option = st.selectbox(
        "Choose ML Model",
        [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "SVM"
        ]
    )

    # ---------------- PREDICT BUTTON ----------------

    if st.button("Predict"):

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=X.columns
        )

        # ---------------- MODEL PREDICTION ----------------

        if model_option == "Logistic Regression":

            prediction = lr_model.predict(input_data)

            probability = lr_model.predict_proba(input_data)

        elif model_option == "Random Forest":

            prediction = rf_model.predict(input_data)

            probability = rf_model.predict_proba(input_data)

        elif model_option == "Decision Tree":

            prediction = dt_model.predict(input_data)

            probability = dt_model.predict_proba(input_data)

        else:

            prediction = svm_model.predict(input_data)

            probability = svm_model.predict_proba(input_data)

        # ---------------- RESULT ----------------

        st.subheader("Prediction Result")

        if prediction[0] == 0:

            st.success("✅ The person is NOT diabetic")

        else:

            st.error("⚠️ The person is diabetic")

        # ---------------- PROBABILITY ----------------

        diabetic_prob = probability[0][1] * 100

        non_diabetic_prob = probability[0][0] * 100

        st.write(
            "### Diabetes Probability:",
            round(diabetic_prob, 2),
            "%"
        )

        # ---------------- PROBABILITY GRAPH ----------------

        st.subheader("Prediction Probability Graph")

        graph_data = pd.DataFrame({
            "Condition": [
                "Non-Diabetic",
                "Diabetic"
            ],
            "Probability": [
                non_diabetic_prob,
                diabetic_prob
            ]
        })

        fig3, ax3 = plt.subplots()

        bars = ax3.bar(
            graph_data["Condition"],
            graph_data["Probability"]
        )

        ax3.set_ylabel("Probability (%)")

        ax3.set_title(
            "Diabetes Prediction Probability"
        )

        # Percentage Labels

        for bar in bars:

            height = bar.get_height()

            ax3.text(
                bar.get_x() + bar.get_width()/2,
                height + 1,
                f'{height:.2f}%',
                ha='center'
            )

        st.pyplot(fig3)

        # ---------------- BMI ANALYSIS ----------------

        st.subheader("BMI Analysis")

        if bmi < 18.5:

            st.info("Underweight")

        elif bmi < 25:

            st.success("Normal Weight")

        elif bmi < 30:

            st.warning("Overweight")

        else:

            st.error("Obese")

        # ---------------- HEALTH SUGGESTIONS ----------------

        st.subheader("Health Suggestions")

        st.write("• Exercise regularly")

        st.write("• Maintain healthy diet")

        st.write("• Monitor glucose levels")

        st.write("• Drink enough water")

        st.write("• Sleep properly")

# ---------------- DATASET PAGE ----------------

elif option == "Dataset":

    st.title("📊 Diabetes Dataset")

    st.write(df)

    st.subheader("Dataset Statistics")

    st.write(df.describe())

# ---------------- GRAPH PAGE ----------------

elif option == "Graphs":

    st.title("📈 Dataset Visualization")

    # ---------------- OUTCOME COUNT ----------------

    st.subheader("Diabetes Outcome Count")

    fig1, ax1 = plt.subplots()

    sns.countplot(
        x='Outcome',
        data=df,
        ax=ax1
    )

    st.pyplot(fig1)

    # ---------------- HEATMAP ----------------

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax2
    )

    st.pyplot(fig2)

    # ---------------- FEATURE IMPORTANCE ----------------

    st.subheader("Feature Importance Graph")

    importance = rf_model.feature_importances_

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    })

    feature_importance = feature_importance.sort_values(
        by='Importance',
        ascending=False
    )

    fig4, ax4 = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance,
        ax=ax4
    )

    ax4.set_title(
        "Feature Importance"
    )

    st.pyplot(fig4)

    # ---------------- FOOD IMPACT GRAPH ----------------

    st.subheader("🍔 Food Items That May Affect Diabetes")

    food_data = pd.DataFrame({

        "Food Item": [
            "Sugary Drinks",
            "White Bread",
            "Fast Food",
            "Sweets",
            "Rice",
            "Processed Snacks",
            "Fried Food",
            "Ice Cream"
        ],

        "Impact Level": [
            95,
            85,
            90,
            92,
            75,
            80,
            88,
            86
        ]
    })

    fig6, ax6 = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x='Impact Level',
        y='Food Item',
        data=food_data,
        ax=ax6
    )

    ax6.set_title(
        "Food Items Affecting Diabetes Risk"
    )

    ax6.set_xlabel(
        "Impact Level (%)"
    )

    ax6.set_ylabel(
        "Food Items"
    )

    st.pyplot(fig6)

    # ---------------- HEALTHY FOOD GRAPH ----------------

    st.subheader("🥗 Recommended Healthy Foods")

    healthy_foods = pd.DataFrame({

        "Healthy Food": [
            "Vegetables",
            "Oats",
            "Brown Rice",
            "Fruits",
            "Nuts",
            "Whole Grains",
            "Salads",
            "Protein-rich Foods"
        ],

        "Health Score": [
            95,
            90,
            88,
            85,
            92,
            89,
            96,
            91
        ]
    })

    fig7, ax7 = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x='Health Score',
        y='Healthy Food',
        data=healthy_foods,
        ax=ax7
    )

    ax7.set_title(
        "Healthy Foods for Diabetes Control"
    )

    ax7.set_xlabel(
        "Health Benefit Score (%)"
    )

    ax7.set_ylabel(
        "Healthy Foods"
    )

    st.pyplot(fig7)