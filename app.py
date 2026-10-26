import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

from pathlib import Path
from datetime import datetime
import uuid




st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)




st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background-color: #f5f7fb;
        color: #172033;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* Sidebar buttons */

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 46px;

        background-color: #1e293b !important;
        color: #ffffff !important;

        border: 1px solid #334155 !important;
        border-radius: 8px;

        font-size: 15px;
        font-weight: 600;

        text-align: center;
        white-space: normal;
        overflow: visible;

        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #273449 !important;
        border-color: #64748b !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stButton > button p {
        color: #ffffff !important;
    }

    /* --------------------------------------------------------
       ALL MAIN BUTTONS
    -------------------------------------------------------- */

    .stButton > button {
        min-height: 44px;

        background-color: #172033 !important;
        color: #ffffff !important;

        border: 1px solid #334155 !important;
        border-radius: 8px;

        font-weight: 600;
        font-size: 14px;

        white-space: normal;
        overflow: visible;

        box-shadow: none !important;
    }

    .stButton > button:hover {
        background-color: #243044 !important;
        border-color: #64748b !important;
        color: #ffffff !important;
    }

    .stButton > button:focus {
        color: #ffffff !important;
        border-color: #64748b !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    .stButton > button p {
        color: #ffffff !important;
    }

    /* --------------------------------------------------------
       HEADINGS
    -------------------------------------------------------- */

    h1 {
        color: #102a56 !important;
        font-weight: 700 !important;
    }

    h2 {
        color: #102a56 !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #183b6b !important;
        font-weight: 650 !important;
    }

    /* --------------------------------------------------------
       LABELS
    -------------------------------------------------------- */

    label,
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        color: #26364d !important;
        font-weight: 600 !important;
    }

    /* --------------------------------------------------------
       INPUT BOXES
    -------------------------------------------------------- */

    .stTextInput input,
    .stNumberInput input {
        color: #172033 !important;
        background-color: #ffffff !important;

        border: 1px solid #cbd5e1 !important;
        border-radius: 7px !important;
    }

    .stTextInput input::placeholder,
    .stNumberInput input::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    /* --------------------------------------------------------
       SELECTBOX
    -------------------------------------------------------- */

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
    }

    div[data-baseweb="select"] span {
        color: #172033 !important;
    }

    /* Dropdown menu */

    div[role="listbox"] {
        background-color: #ffffff !important;
    }

    div[role="option"] {
        color: #172033 !important;
        background-color: #ffffff !important;
    }

    div[role="option"]:hover {
        background-color: #eaf1fb !important;
        color: #102a56 !important;
    }

    /* --------------------------------------------------------
       CHECKBOX
    -------------------------------------------------------- */

    .stCheckbox label {
        color: #26364d !important;
    }

    /* --------------------------------------------------------
       CAPTIONS
    -------------------------------------------------------- */

    .stCaption {
        color: #64748b !important;
    }

    /* --------------------------------------------------------
       INFO / WARNING / SUCCESS / ERROR
    -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    div[data-testid="stAlert"] p {
        color: #172033 !important;
    }

    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* --------------------------------------------------------
       EXPANDERS
    -------------------------------------------------------- */

    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #dbe2ea;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] summary {
        color: #172033 !important;
        font-weight: 600 !important;
    }

    /* --------------------------------------------------------
       METRIC
    -------------------------------------------------------- */

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #102a56 !important;
    }

    /* --------------------------------------------------------
       DIVIDER
    -------------------------------------------------------- */

    hr {
        border-color: #dbe2ea !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "credit_risk_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def get_shap_explainer():
    classifier = model.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


explainer = get_shap_explainer()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "selected_assessment" not in st.session_state:
    st.session_state.selected_assessment = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_assessment_id():
    return (
        "CR-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + str(uuid.uuid4())[:4].upper()
    )


def calculate_risk(probability):

    LOW_THRESHOLD = 0.04
    HIGH_THRESHOLD = 0.32

    if probability <= LOW_THRESHOLD:
        return "Low"

    elif probability <= HIGH_THRESHOLD:
        return "Medium"

    return "High"


def get_risk_icon(category):

    if category == "Low":
        return "🟢"

    elif category == "Medium":
        return "🟡"

    return "🔴"


def get_risk_description(category):

    if category == "Low":
        return (
            "The applicant has a relatively low predicted probability "
            "of loan default."
        )

    elif category == "Medium":
        return (
            "The applicant falls within the moderate predicted default "
            "risk range and may require additional review."
        )

    return (
        "The applicant has a relatively high predicted probability "
        "of loan default and requires careful review."
    )


def get_feature_display_name(feature):

    feature = str(feature)

    replacements = {
        "num__person_age": "Applicant Age",
        "num__person_income": "Annual Income",
        "num__person_emp_length": "Employment Length",
        "num__loan_amnt": "Loan Amount",
        "num__loan_int_rate": "Interest Rate",
        "num__loan_percent_income": "Loan-to-Income Ratio",
        "num__cb_person_cred_hist_length": "Credit History Length",

        "cat__person_home_ownership_RENT": "Home Ownership: Rent",
        "cat__person_home_ownership_OWN": "Home Ownership: Own",
        "cat__person_home_ownership_MORTGAGE": "Home Ownership: Mortgage",
        "cat__person_home_ownership_OTHER": "Home Ownership: Other",

        "cat__loan_intent_PERSONAL": "Loan Intent: Personal",
        "cat__loan_intent_EDUCATION": "Loan Intent: Education",
        "cat__loan_intent_MEDICAL": "Loan Intent: Medical",
        "cat__loan_intent_VENTURE": "Loan Intent: Venture",
        "cat__loan_intent_HOMEIMPROVEMENT": "Loan Intent: Home Improvement",
        "cat__loan_intent_DEBTCONSOLIDATION": "Loan Intent: Debt Consolidation",

        "cat__loan_grade_A": "Loan Grade: A",
        "cat__loan_grade_B": "Loan Grade: B",
        "cat__loan_grade_C": "Loan Grade: C",
        "cat__loan_grade_D": "Loan Grade: D",
        "cat__loan_grade_E": "Loan Grade: E",
        "cat__loan_grade_F": "Loan Grade: F",
        "cat__loan_grade_G": "Loan Grade: G",

        "cat__cb_person_default_on_file_Y": "Previous Default: Yes",
        "cat__cb_person_default_on_file_N": "Previous Default: No",
    }

    if feature in replacements:
        return replacements[feature]

    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")

    return feature.replace("_", " ").title()


def get_shap_explanation(applicant):

    preprocessor = model.named_steps["preprocessor"]

    transformed = preprocessor.transform(applicant)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    shap_values = explainer.shap_values(transformed)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            values = shap_values[1][0]

        else:
            values = shap_values[0][0]

    else:

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:

            # shape: samples, features, classes
            values = shap_values[0, :, 1]

        elif shap_values.ndim == 2:

            values = shap_values[0]

        else:

            values = shap_values

    explanation_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": values
        }
    )

    explanation_df["Absolute Impact"] = explanation_df[
        "SHAP Value"
    ].abs()

    explanation_df = explanation_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    return explanation_df.head(5)


def convert_numpy_values(data):

    converted = {}

    for key, value in data.items():

        if isinstance(value, np.generic):
            converted[key] = value.item()

        else:
            converted[key] = value

    return converted


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏦 Credit Risk")

    st.caption("Intelligence & Assessment Platform")

    st.divider()

    new_assessment_clicked = st.button(
        "＋  New Assessment",
        key="new_assessment_button",
        width="stretch"
    )

    if new_assessment_clicked:
        st.session_state.selected_assessment = None
        st.rerun()

    st.markdown("### Assessment History")

    # --------------------------------------------------------
    # Search history
    # --------------------------------------------------------

    if len(st.session_state.history) > 0:

        search_history = st.text_input(
            "Search",
            placeholder="Applicant name or Customer ID",
            key="history_search"
        )

        search_term = search_history.strip().lower()

        filtered_history = []

        for item in st.session_state.history:

            name = str(item["applicant_name"]).lower()
            customer_id = str(item["customer_id"]).lower()
            assessment_id = str(item["assessment_id"]).lower()

            if (
                search_term == ""
                or search_term in name
                or search_term in customer_id
                or search_term in assessment_id
            ):
                filtered_history.append(item)

        if len(filtered_history) == 0:

            st.caption("No matching assessments found.")

        else:

            for item in reversed(filtered_history):

                icon = get_risk_icon(item["risk_category"])

                button_text = (
                    f"{icon}  {item['applicant_name']}"
                )

                clicked = st.button(
                    button_text,
                    key=f"history_{item['assessment_id']}",
                    width="stretch"
                )

                st.caption(
                    f"{item['customer_id']}  •  "
                    f"{item['risk_category']}  •  "
                    f"{item['probability']:.2f}%"
                )

                if clicked:

                    st.session_state.selected_assessment = item
                    st.rerun()

    else:

        st.caption("No assessments yet.")

    st.divider()

    st.caption("Internal Risk Assessment")

    st.caption("ML Model • Explainable AI")


# ============================================================
# MAIN PAGE
# ============================================================

st.title("Credit Risk Intelligence")

st.caption(
    "Enterprise machine learning platform for explainable credit risk assessment."
)


# ============================================================
# HISTORICAL ASSESSMENT VIEW
# ============================================================

if st.session_state.selected_assessment is not None:

    assessment = st.session_state.selected_assessment

    st.subheader("Assessment Details")

    st.caption(
        f"Assessment ID: {assessment['assessment_id']}  •  "
        f"{assessment['timestamp']}"
    )

    st.divider()

    # --------------------------------------------------------
    # Applicant heading
    # --------------------------------------------------------

    st.subheader(
        f"Applicant: {assessment['applicant_name']}"
    )

    st.write(
        f"**Customer / Application ID:** "
        f"{assessment['customer_id']}"
    )

    st.divider()

    # --------------------------------------------------------
    # Risk Summary
    # --------------------------------------------------------

    st.subheader("Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Risk Category",
            f"{get_risk_icon(assessment['risk_category'])} "
            f"{assessment['risk_category']}"
        )

    with col2:

        st.metric(
            "Risk Probability",
            f"{assessment['probability']:.2f}%"
        )

    with col3:

        st.metric(
            "Risk Score",
            f"{assessment['risk_score']:.2f} / 100"
        )

    st.info(
        get_risk_description(
            assessment["risk_category"]
        )
    )

    st.divider()

    # --------------------------------------------------------
    # Applicant Information
    # --------------------------------------------------------

    st.subheader("Applicant Information")

    applicant_data = assessment["applicant"]

    display_data = {
        "Applicant Age": applicant_data.get("person_age"),
        "Annual Income": applicant_data.get("person_income"),
        "Home Ownership": applicant_data.get("person_home_ownership"),
        "Employment Length": applicant_data.get("person_emp_length"),
        "Loan Intent": applicant_data.get("loan_intent"),
        "Loan Grade": applicant_data.get("loan_grade"),
        "Loan Amount": applicant_data.get("loan_amnt"),
        "Interest Rate": applicant_data.get("loan_int_rate"),
        "Loan-to-Income Ratio": applicant_data.get(
            "loan_percent_income"
        ),
        "Previous Default": applicant_data.get(
            "cb_person_default_on_file"
        ),
        "Credit History Length": applicant_data.get(
            "cb_person_cred_hist_length"
        ),
    }

    display_df = pd.DataFrame(
        [display_data]
    )

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------------
    # Explainable AI
    # --------------------------------------------------------

    st.subheader("Explainable AI")

    st.caption(
        "Top factors influencing this risk prediction."
    )

    for factor in assessment["top_factors"]:

        feature_name = factor["feature"]
        impact = factor["impact"]

        if impact >= 0:

            st.markdown(
                f"🔴 **{feature_name}** — "
                f"increases predicted default risk"
            )

        else:

            st.markdown(
                f"🟢 **{feature_name}** — "
                f"decreases predicted default risk"
            )

        st.write("")

    st.divider()

    if st.button(
        "←  Back to New Assessment",
        key="back_to_new_assessment",
        width="content"
    ):

        st.session_state.selected_assessment = None
        st.rerun()

    st.stop()


# ============================================================
# NEW ASSESSMENT
# ============================================================

st.subheader("Applicant Identification")

col1, col2 = st.columns(2)

with col1:

    applicant_name = st.text_input(
        "Applicant Name *",
        placeholder="Enter full name"
    )

with col2:

    customer_id = st.text_input(
        "Customer / Application ID *",
        placeholder="Example: CUST-10245"
    )


st.divider()


# ============================================================
# APPLICANT PROFILE
# ============================================================

st.subheader("Applicant Profile")

col1, col2, col3 = st.columns(3)

with col1:

    person_age = st.number_input(
        "Applicant Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:

    person_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000,
        step=1000
    )

with col3:

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER"
        ]
    )


col1, col2, col3 = st.columns(3)

with col1:

    person_emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

with col2:

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

with col3:

    cb_person_default_on_file = st.selectbox(
        "Previous Default on File",
        [
            "N",
            "Y"
        ]
    )


# ============================================================
# LOAN DETAILS
# ============================================================

st.subheader("Loan Details")

col1, col2, col3 = st.columns(3)

with col1:

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

with col2:

    loan_grade = st.selectbox(
        "Loan Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ]
    )

with col3:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000,
        step=500
    )


col1, col2 = st.columns(2)

with col1:

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.5,
        step=0.1
    )

with col2:

    loan_percent_income = st.number_input(
        "Loan Percent of Income",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        format="%.2f"
    )


st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.info(
    "The model uses a tuned Random Forest classifier. "
    "The prediction probability is converted into a 0–100 risk score "
    "and then categorized into Low, Medium, or High risk."
)


# ============================================================
# RUN ASSESSMENT
# ============================================================

run_assessment = st.button(
    "Run Credit Risk Assessment",
    key="run_assessment",
    width="stretch"
)


if run_assessment:

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if applicant_name.strip() == "":

        st.error(
            "Please enter the Applicant Name."
        )

        st.stop()

    if customer_id.strip() == "":

        st.error(
            "Please enter the Customer / Application ID."
        )

        st.stop()

    # --------------------------------------------------------
    # Prepare customer/application ID
    # --------------------------------------------------------

    customer_id = customer_id.strip().upper()

    applicant_name = applicant_name.strip()

    # --------------------------------------------------------
    # Create model input
    # --------------------------------------------------------

    applicant = pd.DataFrame(
        [
            {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": person_home_ownership,
                "person_emp_length": person_emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_person_default_on_file,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
            }
        ]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(applicant)[0][1]
    )

    risk_score = probability * 100

    risk_category = calculate_risk(
        probability
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    try:

        top_factors = get_shap_explanation(
            applicant
        )

    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated for this assessment."
        )

        top_factors = pd.DataFrame(
            columns=[
                "Feature",
                "SHAP Value",
                "Absolute Impact"
            ]
        )

    # --------------------------------------------------------
    # Convert applicant data for session storage
    # --------------------------------------------------------

    applicant_data = convert_numpy_values(
        applicant.iloc[0].to_dict()
    )

    # --------------------------------------------------------
    # Convert SHAP factors
    # --------------------------------------------------------

    top_factors_list = []

    for _, row in top_factors.iterrows():

        top_factors_list.append(
            {
                "feature": get_feature_display_name(
                    row["Feature"]
                ),
                "impact": float(
                    row["SHAP Value"]
                )
            }
        )

    # --------------------------------------------------------
    # Create assessment object
    # --------------------------------------------------------

    assessment = {

        "applicant_name": applicant_name,

        "customer_id": customer_id,

        "assessment_id": create_assessment_id(),

        "timestamp": datetime.now().strftime(
            "%d %b %Y, %I:%M %p"
        ),

        "applicant": applicant_data,

        "probability": risk_score,

        "risk_score": risk_score,

        "risk_category": risk_category,

        "top_factors": top_factors_list
    }

    # --------------------------------------------------------
    # Save to session history
    # --------------------------------------------------------

    st.session_state.history.append(
        assessment
    )

    st.session_state.selected_assessment = assessment

    st.rerun()