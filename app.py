import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import plotly.figure_factory as ff


st.set_page_config(
    page_title="MLP-RFE Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "mlp_rfe_pipeline.joblib"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "mlp_rfe_metadata.json"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


model = load_model()
metadata = load_metadata()


st.sidebar.title("🫀 MLP-RFE")
st.sidebar.caption("Predictive Modelling Dashboard")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("### Navigation")

st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        justify-content: flex-start;
        padding-left: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button(
    "🏠  Overview",
    use_container_width=True
):
    st.session_state["page"] = "🏠 Overview"

if st.sidebar.button(
    "📊  Model Performance",
    use_container_width=True
):
    st.session_state["page"] = "📊 Model Performance"

if st.sidebar.button(
    "🎯  RFE Feature Selection",
    use_container_width=True
):
    st.session_state["page"] = "🎯 RFE Feature Selection"

if st.sidebar.button(
    "🩺  Patient Prediction",
    use_container_width=True
):
    st.session_state["page"] = "🩺 Patient Prediction"

page = st.session_state.get(
    "page",
    "🏠 Overview"
)



st.sidebar.divider()

st.sidebar.caption("RESEARCH PROJECT")

st.sidebar.write("**ⓘ UniSZA/2024/PGK/03**")

feature_names = {
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "exang": "Exercise-Induced Angina",
    "slope": "ST Segment Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia"
}


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.title("MLP-RFE Heart Disease Prediction")

    st.markdown("### Research Demonstration Dashboard")

    st.markdown(
        """
        This dashboard demonstrates the **Multi-Layer Perceptron-
        Recursive Feature Elimination (MLP-RFE)** model developed
        for heart disease prediction using the Cleveland Heart
        Disease dataset.
        """
    )

    st.divider()

    st.header("Research Project")

    project_col1, project_col2 = st.columns([1, 2])

    with project_col1:

        st.markdown("#### Project Code")
        st.info("**UniSZA/2024/PGK/03**")

        st.markdown("#### Research Cluster")
        st.info("**Information and Communication Technology**")

    with project_col2:

        st.markdown("#### Official Research Title")

        st.markdown(
            """
            **Improved Multilayer Perceptron-Recursive Feature
            Elimination (MLP-RFE) Model Based on Deep Learning
            Algorithms in Enhancing Heart Disease Prediction’s Accuracy**
            """
        )

    st.divider()

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Records", "303")

    with col2:
        st.metric("Original Features", "13")

    with col3:
        st.metric(
            "RFE-Selected Features",
            "8",
            delta="5 fewer features"
        )

    with col4:
        st.metric(
            "MLP-RFE Test Accuracy",
            "85.25%"
        )

    st.divider()

    st.header("Model Development Workflow")

    st.caption(
        "Overview of the existing research workflow from the "
        "Cleveland dataset to the final MLP-RFE prediction model."
    )

    workflow1, workflow2, workflow3, workflow4 = st.columns(4)

    with workflow1:

        st.markdown("### ① Dataset")

        st.markdown(
            """
            **Cleveland Heart Disease**

            303 patient records

            13 original input features
            """
        )

    with workflow2:

        st.markdown("### ② RFE")

        st.markdown(
            """
            **Feature Selection**

            13 input features → **8 selected features**
            """
        )

    with workflow3:

        st.markdown("### ③ MLP")

        st.markdown(
            """
            **Neural Network Classifier**

            Trained using the RFE-selected features
            """
        )

    with workflow4:

        st.markdown("### ④ Prediction")

        st.markdown(
            """
            **Binary Classification**

            Class 0: No heart disease

            Class 1: Heart disease
            """
        )

    st.divider()

    # ============================================================
    # RFE-SELECTED FEATURES
    # ============================================================

    st.header("RFE-Selected Features")

    st.write(
        "The final MLP-RFE classification pipeline uses eight features "
        "selected from the original 13-feature representation."
    )

    selected_features = metadata["selected_features"]

    st.markdown("### 8 Selected Features")

    feature_rows = [
        selected_features[:4],
        selected_features[4:]
    ]

    for row in feature_rows:

        feature_cols = st.columns(4)

        for i, feature in enumerate(row):

            with feature_cols[i]:

                display_name = feature_names.get(
                    feature,
                    feature
                )

                st.markdown(
                    f"**{display_name}**"
                )

                st.caption(
                    f"`{feature}`"
                )

    st.caption(
        "These eight features are used as inputs to the final "
        "MLP-RFE classification pipeline."
    )

    st.divider()

    st.header("Model Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.markdown("#### MLP Classifier")

        st.write(
            """
            **Algorithm:** Multi-Layer Perceptron

            **Architecture:** 64 → 32 hidden units

            **Activation:** ReLU

            **Solver:** Adam

            **Maximum iterations:** 500
            """
        )

    with info_col2:

        st.markdown("#### RFE Configuration")

        st.write(
            """
            **Method:** Recursive Feature Elimination

            **Estimator:** Logistic Regression

            **Target:** 8 selected features

            **Feature reduction:** 13 → 8
            """
        )

    st.divider()

    st.header("Model Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:

        st.success(
            "✓ Trained MLP-RFE pipeline loaded successfully"
        )

    with status_col2:

        st.success(
            "✓ Dashboard uses the saved model without retraining"
        )

    st.divider()

    st.caption(
        """
        **Research Demonstration Only:** This dashboard is intended
        for research and educational demonstration purposes. It is
        not a clinical diagnostic tool and should not be used as a
        substitute for professional medical assessment or treatment.
        """
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title("Model Performance")
    st.markdown(
        """
        Performance results comparing the original baseline MLP model
        using all 13 input features with the proposed MLP-RFE model
        using the 8 features selected by Recursive Feature Elimination.
        """
    )

    st.divider()

    # ============================================================
    # PERFORMANCE COMPARISON
    # ============================================================

    st.header("Performance Comparison")

    st.markdown(
        """
        The baseline MLP provides the reference point for evaluating
        the effect of RFE-based feature selection. Both models use the
        same dataset and evaluation setting, while the MLP-RFE model
        uses a reduced set of input features.
        """
    )

    comparison_data = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall (Sensitivity)",
            "F1-score",
            "ROC-AUC",
            "Input Features"
        ],
        "Baseline MLP": [
            "83.6%",
            "75.0%",
            "96.4%",
            "84.4%",
            "93.4%",
            "13"
        ],
        "MLP-RFE": [
            "85.2%",
            "80.6%",
            "89.3%",
            "84.7%",
            "91.5%",
            "8"
        ]
    })

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Results are based on the held-out test set used during model evaluation."
    )

    st.divider()

    # ============================================================
    # UNDERSTANDING THE EVALUATION METRICS
    # ============================================================

    st.header("Understanding the Evaluation Metrics")

    st.markdown(
        """
        Multiple evaluation metrics are reported because no single
        metric fully describes binary classification performance.
        Each metric highlights a different aspect of the model's
        predictive behaviour.
        """
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.markdown("#### Accuracy")
        st.write(
            """
            Accuracy measures the proportion of all test cases that
            were classified correctly. It provides an overall measure
            of classification performance.
            """
        )

        st.markdown("#### Precision")
        st.write(
            """
            Precision measures the proportion of cases predicted as
            having heart disease that were actually positive. Higher
            precision indicates fewer false-positive predictions.
            """
        )

        st.markdown("#### Recall (Sensitivity)")
        st.write(
            """
            Recall measures the proportion of actual heart-disease
            cases that were correctly identified. Higher recall means
            fewer positive cases are missed by the model.
            """
        )

    with metric_col2:

        st.markdown("#### F1-score")
        st.write(
            """
            F1-score is the harmonic mean of precision and recall.
            It provides a balanced measure when both types of
            classification performance are important.
            """
        )

        st.markdown("#### ROC-AUC")
        st.write(
            """
            ROC-AUC measures the model's ability to distinguish
            between the two classes across different classification
            thresholds. A higher value indicates stronger overall
            class discrimination.
            """
        )

    st.divider()

    # ============================================================
    # MLP-RFE PERFORMANCE
    # ============================================================

    st.header("MLP-RFE Performance")

    st.markdown(
        """
        The proposed MLP-RFE model achieved the following results
        on the held-out test set.
        """
    )

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:
        st.metric("Accuracy", "85.2%")

    with metric2:
        st.metric("Precision", "80.6%")

    with metric3:
        st.metric("Recall", "89.3%")

    with metric4:
        st.metric("F1-score", "84.7%")

    with metric5:
        st.metric("ROC-AUC", "91.5%")

    st.info(
        """
        **Key observation:** MLP-RFE achieved higher accuracy,
        precision, and F1-score than the baseline MLP, while using
        only 8 instead of 13 input features.
        """
    )

    st.divider()

    # ============================================================
    # CONFUSION MATRICES
    # ============================================================

    st.header("Confusion Matrices")

    st.markdown(
        """
        A confusion matrix provides a detailed view of the model's
        classification outcomes. It separates predictions into four
        categories: true negatives (TN), false positives (FP),
        false negatives (FN), and true positives (TP).
        """
    )

    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:

        st.subheader("Baseline MLP")

        baseline_cm = np.array([
            [24, 9],
            [1, 27]
        ])

        st.write(
            """
            **Confusion Matrix**
            """
        )

        st.dataframe(
            pd.DataFrame(
                baseline_cm,
                index=["Actual: No Disease", "Actual: Disease"],
                columns=["Predicted: No Disease", "Predicted: Disease"]
            ),
            use_container_width=True
        )

        st.markdown(
            """
            **Interpretation**

            - **24 TN:** No-disease cases correctly classified.
            - **9 FP:** No-disease cases incorrectly classified as disease.
            - **1 FN:** Disease case incorrectly classified as no disease.
            - **27 TP:** Disease cases correctly classified.
            """
        )

        st.info(
            """
            The baseline model correctly identified 27 of the
            28 actual disease cases. This contributes to its high
            recall of **96.4%**, although it also produced
            **9 false-positive predictions**.
            """
        )

    with cm_col2:

        st.subheader("MLP-RFE")

        mlp_rfe_cm = np.array([
            [27, 6],
            [3, 25]
        ])

        st.write(
            """
            **Confusion Matrix**
            """
        )

        st.dataframe(
            pd.DataFrame(
                mlp_rfe_cm,
                index=["Actual: No Disease", "Actual: Disease"],
                columns=["Predicted: No Disease", "Predicted: Disease"]
            ),
            use_container_width=True
        )

        st.markdown(
            """
            **Interpretation**

            - **27 TN:** No-disease cases correctly classified.
            - **6 FP:** No-disease cases incorrectly classified as disease.
            - **3 FN:** Disease cases incorrectly classified as no disease.
            - **25 TP:** Disease cases correctly classified.
            """
        )

        st.info(
            """
            MLP-RFE reduced false-positive predictions from **9 to 6**,
            supporting its higher precision of **80.6%**. However,
            false negatives increased from **1 to 3**, resulting in
            lower recall of **89.3%**.
            """
        )

    st.divider()

    # ============================================================
    # KEY FINDINGS
    # ============================================================

    st.header("Key Findings")

    finding_col1, finding_col2 = st.columns(2)

    with finding_col1:

        st.markdown("#### Baseline MLP")

        st.markdown(
            """
            - Achieved strong overall performance.
            - Demonstrated very high recall (**96.4%**).
            - Produced fewer false negatives (**1**).
            - Precision was lower (**75.0%**), reflecting more
              false-positive predictions.
            """
        )

    with finding_col2:

        st.markdown("#### MLP-RFE")

        st.markdown(
            """
            - Accuracy improved from **83.6% to 85.2%**.
            - Precision improved from **75.0% to 80.6%**.
            - F1-score improved slightly from **84.4% to 84.7%**.
            - Input dimensionality was reduced from **13 to 8 features**.
            """
        )

    st.info(
        """
        **Feature reduction:** RFE reduced the model input
        dimensionality by approximately **38%**, from 13 original
        features to 8 selected features.
        """
    )

    st.divider()

    # ============================================================
    # RESEARCH SIGNIFICANCE
    # ============================================================

    st.header("Research Significance")

    st.markdown(
        """
        The results demonstrate the potential value of feature
        selection when developing deep-learning-based prediction
        models. In this study, the observed performance changes
        were achieved without modifying the MLP architecture itself.
        """
    )

    significance_col1, significance_col2, significance_col3 = st.columns(3)

    with significance_col1:
        st.markdown("#### Feature Relevance")
        st.write(
            """
            Highlights the potential value of selecting relevant
            features before model training.
            """
        )

    with significance_col2:
        st.markdown("#### Model Efficiency")
        st.write(
            """
            Demonstrates that a reduced feature representation can
            maintain competitive predictive performance.
            """
        )

    with significance_col3:
        st.markdown("#### Practical Application")
        st.write(
            """
            Provides a practical feature-reduction approach for
            heart disease prediction research.
            """
        )

    st.divider()

    # ============================================================
    # INTERPRETATION
    # ============================================================

    st.header("Interpretation")

    st.markdown(
        """
        Compared with the baseline MLP, the MLP-RFE model achieved
        higher accuracy, precision, and F1-score while reducing the
        number of input features from 13 to 8.

        However, recall and ROC-AUC were lower for the MLP-RFE model.
        Therefore, the MLP-RFE model should not be interpreted as
        uniformly superior. Instead, it provides a more compact
        feature representation with a different performance profile.

        The baseline model demonstrates strong sensitivity but lower
        precision, while MLP-RFE reduces false-positive predictions
        and improves precision at the expense of some recall.
        """
    )

    st.success(
        """
        **Conclusion:** Feature selection shifted the model from a
        highly sensitivity-driven performance profile toward a more
        balanced classification profile, with improvements in
        accuracy, precision, and F1-score alongside reduced recall
        and ROC-AUC.
        """
    )

    st.divider()

    st.caption(
        """
        **Research Demonstration Only:** This dashboard is intended
        for research and educational demonstration purposes. It is
        not a clinical diagnostic tool and should not be used as a
        substitute for professional medical assessment or treatment.
        """
    )

# ============================================================
# RFE FEATURE SELECTION
# ============================================================

elif page == "🎯 RFE Feature Selection":

    st.title("RFE Feature Selection")

    st.markdown(
        """
        Recursive Feature Elimination (RFE) was applied to the original
        13 input features to identify a reduced set of 8 features for
        the MLP-RFE model.

        The RFE procedure used **Logistic Regression** as the estimator
        and selected **8 features** from the original 13.
        """
    )

    st.divider()

    # ============================================================
    # WHY FEATURE SELECTION?
    # ============================================================

    st.header("Why Feature Selection?")

    st.markdown(
        """
        Feature selection is used to reduce the number of input
        variables presented to a predictive model while retaining
        features identified as relevant by the selection procedure.
        This can be particularly useful when working with clinical
        and tabular datasets containing variables with different
        levels of predictive relevance.
        """
    )

    reason_col1, reason_col2 = st.columns(2)

    with reason_col1:

        st.markdown("#### Potential Challenges")

        st.markdown(
            """
            ✓ Clinical datasets can contain redundant or less
            informative features.

            ✓ An unfiltered feature space may introduce unnecessary
            complexity and reduce model efficiency.
            """
        )

    with reason_col2:

        st.markdown("#### Role of Feature Selection")

        st.markdown(
            """
            ✓ Reduces the number of variables used by the model.

            ✓ Provides a more compact input representation.

            ✓ Focuses model development on the features retained
            by the selection procedure.
            """
        )

    st.divider()

    # ============================================================
    # WHY RFE?
    # ============================================================

    st.header("Why RFE Is Applied")

    st.markdown(
        """
        Recursive Feature Elimination provides a systematic approach
        for reducing the feature set. It repeatedly fits the selected
        estimator, identifies the least influential feature according
        to the estimator, and eliminates features until the desired
        number of variables remains.
        """
    )

    rfe_col1, rfe_col2, rfe_col3 = st.columns(3)

    with rfe_col1:

        st.markdown("#### ① Iterative Evaluation")

        st.write(
            """
            Evaluates feature relevance using the selected
            Logistic Regression estimator.
            """
        )

    with rfe_col2:

        st.markdown("#### ② Feature Elimination")

        st.write(
            """
            Eliminates less influential features based on the
            magnitude of the estimator's coefficients.
            """
        )

    with rfe_col3:

        st.markdown("#### ③ Compact Representation")

        st.write(
            """
            Produces a reduced feature representation for
            subsequent MLP model development.
            """
        )

    st.divider()

    # ============================================================
    # FEATURE REDUCTION
    # ============================================================

    st.header("Feature Reduction")

    reduction_col1, reduction_col2, reduction_col3, reduction_col4 = st.columns(4)

    with reduction_col1:
        st.metric("Original Features", "13")

    with reduction_col2:
        st.metric("Selected Features", "8")

    with reduction_col3:
        st.metric("Features Removed", "5")

    with reduction_col4:
        st.metric("Dimensionality Reduction", "38.5%")

    st.info(
        """
        RFE reduced the model input dimensionality from **13 to 8
        features**, removing **5 features** from the original
        representation.
        """
    )

    st.divider()

    # ============================================================
    # SELECTED FEATURES
    # ============================================================

    st.header("Selected Features")

    st.markdown(
        """
        The following eight features were retained by RFE and
        subsequently used as inputs to the MLP-RFE classification
        pipeline.
        """
    )

    selected_features = metadata["selected_features"]

    feature_rows = [
        selected_features[:4],
        selected_features[4:]
    ]

    feature_number = 1

    for row in feature_rows:

        feature_cols = st.columns(4)

        for i, feature in enumerate(row):

            with feature_cols[i]:

                display_name = feature_names.get(feature, feature)

                st.markdown(
                    f"### {feature_number}. {display_name}"
                )

                st.caption(
                    f"Dataset code: `{feature}`"
                )

            feature_number += 1

    st.caption(
        """
        The displayed names are provided for readability. The
        original dataset feature codes remain unchanged.
        """
    )

    st.divider()

    # ============================================================
    # HOW RFE SELECTS FEATURES
    # ============================================================

    st.header("How RFE Selects Features")

    st.markdown(
        """
        **Logistic Regression estimates a coefficient for each feature.
        RFE uses the absolute coefficient magnitudes to identify the
        least influential feature in the fitted model. The identified
        feature is removed, the model is refitted, and the process is
        repeated until the desired number of features remains.**
        """
    )

    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:

        st.markdown("### ① Estimate")

        st.write(
            """
            Logistic Regression assigns coefficients to the features
            in the fitted model.
            """
        )

    with step_col2:

        st.markdown("### ② Eliminate")

        st.write(
            """
            The feature with the smallest absolute coefficient
            magnitude is identified for elimination.
            """
        )

    with step_col3:

        st.markdown("### ③ Repeat")

        st.write(
            """
            The model is refitted and the elimination process
            continues until **8 features remain**.
            """
        )

    st.divider()

    # ============================================================
    # RFE FEATURE RANKING
    # ============================================================

    st.header("RFE Feature Ranking")

    ranking_rows = [
        ["sex", "Sex", 1, "Selected"],
        ["cp", "Chest Pain Type", 1, "Selected"],
        ["restecg", "Resting ECG", 1, "Selected"],
        ["fbs", "Fasting Blood Sugar", 1, "Selected"],
        ["thal", "Thalassemia", 1, "Selected"],
        ["ca", "Number of Major Vessels", 1, "Selected"],
        ["slope", "ST Segment Slope", 1, "Selected"],
        ["exang", "Exercise-Induced Angina", 1, "Selected"],
        ["oldpeak", "ST Depression", 2, "Removed"],
        ["trestbps", "Resting Blood Pressure", 3, "Removed"],
        ["thalach", "Maximum Heart Rate", 4, "Removed"],
        ["age", "Age", 5, "Removed"],
        ["chol", "Cholesterol", 6, "Removed"]
    ]

    ranking_data = pd.DataFrame(
        ranking_rows,
        columns=[
            "Feature",
            "Feature Name",
            "RFE Rank",
            "Selection"
        ]
    )

    st.dataframe(
        ranking_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ============================================================
    # HOW TO INTERPRET THE RANKING
    # ============================================================

    st.header("How to Interpret the Ranking")

    st.markdown(
        """
        An RFE ranking of **1** indicates that the feature was
        retained in the final selected feature set. Higher ranking
        values indicate that the feature was eliminated earlier
        during the recursive feature elimination process.
        """
    )

    st.warning(
        """
        **Important:** The RFE ranking should not be interpreted as
        a direct measure of feature importance. All eight selected
        features have a ranking of **1** because they were retained
        in the final eight-feature subset.
        """
    )

    st.divider()

    # ============================================================
    # CLINICAL RELEVANCE CLARIFICATION
    # ============================================================

    st.header("Clinical Relevance Clarification")

    st.info(
        """
        **Important:** The exclusion of the remaining five features
        should not be interpreted as evidence that they are clinically
        unimportant. Their exclusion reflects the feature-selection
        outcome for this dataset and model-development process.
        For example, age and cholesterol remain clinically relevant
        variables.
        """
    )

    st.divider()

    # ============================================================
    # RESEARCH DISCLAIMER
    # ============================================================

    st.caption(
        """
        **Research Demonstration Only:** This dashboard is intended
        for research and educational demonstration purposes. It is
        not a clinical diagnostic tool and should not be used as a
        substitute for professional medical assessment or treatment.
        """
    )

# ============================================================
# PATIENT PREDICTION
# ============================================================

elif page == "🩺 Patient Prediction":

    st.title("Patient Prediction")

    st.markdown(
        """
        Enter values for the eight features selected by RFE.
        The saved MLP-RFE pipeline will generate a prediction using
        the same preprocessing and trained model used during evaluation.
        """
    )

    st.divider()

    # ============================================================
    # INPUT FEATURES
    # ============================================================

    st.header("Input Features")

    st.caption(
        "Select the appropriate encoded value for each RFE-selected feature. "
        "The displayed labels are provided for readability."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: {
                0: "Female",
                1: "Male"
            }[x],
            help="Dataset encoding: 0 = Female, 1 = Male"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical angina",
                2: "Atypical angina",
                3: "Non-anginal pain",
                4: "Asymptomatic"
            }[x],
            help="Dataset encoding: 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic"
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar",
            options=[0, 1],
            format_func=lambda x: {
                0: "No (≤ 120 mg/dL)",
                1: "Yes (> 120 mg/dL)"
            }[x],
            help="Dataset encoding: 0 = No, 1 = Yes"
        )

        restecg = st.selectbox(
            "Resting ECG",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy"
            }[x],
            help="Dataset encoding: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy"
        )

    with input_col2:

        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes"
            }[x],
            help="Dataset encoding: 0 = No, 1 = Yes"
        )

        slope = st.selectbox(
            "ST Segment Slope",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping"
            }[x],
            help="Dataset encoding: 1 = Upsloping, 2 = Flat, 3 = Downsloping"
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} vessel(s)",
            help="Dataset encoding: 0–3"
        )

        thal = st.selectbox(
            "Thalassemia",
            options=[3, 6, 7],
            format_func=lambda x: {
                3: "Normal",
                6: "Fixed defect",
                7: "Reversible defect"
            }[x],
            help="Dataset encoding: 3 = Normal, 6 = Fixed defect, 7 = Reversible defect"
        )

    st.divider()

    # ============================================================
    # PREDICTION
    # ============================================================

    if st.button(
        "Generate Prediction",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "sex": [sex],
                "cp": [cp],
                "fbs": [fbs],
                "restecg": [restecg],
                "exang": [exang],
                "slope": [slope],
                "ca": [ca],
                "thal": [thal]
            }
        )

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        predicted_probability = probability[int(prediction)]

        st.subheader("Prediction Result")

        if prediction == 0:

            st.success(
                "Predicted Class: 0 — No heart disease"
            )

        else:

            st.error(
                "Predicted Class: 1 — Heart disease"
            )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Class",
                str(int(prediction))
            )

        with result_col2:

            st.metric(
                "Model Probability",
                f"{predicted_probability:.2%}"
            )

        st.caption(
            "The probability shown represents the model's predicted probability "
            "for the selected class. It is not a clinical risk estimate."
        )

        st.info(
            "This prediction is generated by the trained MLP-RFE model "
            "using the same preprocessing and model pipeline used during evaluation. "
            "It is provided for research demonstration only."
        )

    st.divider()

    # ============================================================
    # RESEARCH DISCLAIMER
    # ============================================================

    st.info(
        """
        **Research Demonstration Only:** This interface is intended
        for research and educational demonstration purposes. It is
        not a clinical diagnostic tool and should not be used as a
        substitute for professional medical assessment or treatment.
        """
    )
