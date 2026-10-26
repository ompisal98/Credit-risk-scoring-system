Credit Risk Scoring Model

1. Project Overview

This project is an end-to-end Credit Risk Scoring System developed
to estimate the probability that a loan applicant may default on a loan.
The system includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Training and comparison of multiple machine-learning models
- Hyperparameter tuning
- Default-probability prediction
- Low / Medium / High risk categorization
- SHAP-based explainability
- Streamlit-based user interface
- Assessment history during the current application session

The final application uses a trained machine-learning pipeline stored
in:
models/credit_risk_model.pkl
Important: This project is intended for educational/prototype
purposes. It should not be used as the sole basis for real lending
decisions without appropriate validation, governance, fairness
testing, security, and regulatory review.

2. System Requirements

To run the project on another computer, the following are required.
Operating System
The project can be run on:
- Windows 10/11
- Linux
- macOS
The commands in this README primarily show the Windows PowerShell
setup.
Software Requirements
Required
1. Python 3.x
2. pip
3. A terminal such as PowerShell / Command Prompt
4. A web browser for the Streamlit interface
Recommended
- Visual Studio Code
- VS Code Python extension
- VS Code Jupyter extension

Jupyter is required only if the training notebook needs to be opened or
executed. It is not required simply to run the already-trained
Streamlit application, provided the trained model file is included.

3. Python Packages

The main Python packages required are:
pandas
numpy
scikit-learn
joblib
shap
streamlit
matplotlib

For notebook-based development, Jupyter support may also be required.
The recommended method is to install dependencies from:
requirements.txt
using:
python -m pip install -r requirements.txt



6. Quick Start --- Run the Application on a New Computer

These are the main steps the receiver should follow.

Step 1 --- Extract the Project ZIP

Extract the ZIP file to any convenient location.
Example:
C:\Users\<username>\Desktop\credit risk scoring model
Do not change the internal folder structure, especially:
models/credit_risk_model.pkl
because app.py loads the model from that location.

Step 2 --- Verify Python

Open PowerShell inside the project folder and run:
python --version
If Python is correctly installed, a Python 3.x version should be
displayed.
Also check pip:
python -m pip --version
If the python command is not recognized, Python must be installed and
added to PATH before continuing.

Step 3 --- Open the Project Directory
Example:
cd "C:\Users\<username>\Desktop\credit risk scoring model"
Replace the path with the actual extracted project location.

Step 4 --- Create a Virtual Environment
Run:
python -m venv .venv
This creates an isolated Python environment specifically for the
project.

Step 5 --- Activate the Virtual Environment

Windows PowerShell
.\.venv\Scripts\Activate.ps1
After successful activation, the terminal should look similar to:
(.venv) PS C:\Users\<username>\Desktop\credit risk scoring model>
Windows Command Prompt
.venv\Scripts\activate.bat
Linux / macOS
source .venv/bin/activate

Step 6 --- Upgrade pip (Recommended)
python -m pip install --upgrade pip

Step 7 --- Install Project Dependencies
Run:
python -m pip install -r requirements.txt
Wait until all packages are installed successfully.
If a requirements.txt file is unavailable, the core packages can be
installed manually:
python -m pip install pandas numpy scikit-learn joblib shap streamlit matplotlib

Step 8 --- Verify the Model File

Before starting the application, confirm that this file exists:
models/credit_risk_model.pkl
Expected structure:
credit risk scoring model/
├── app.py
└── models/
    └── credit_risk_model.pkl
If the model file is missing, the Streamlit application cannot perform
predictions.

Step 9 --- Run the Streamlit Application

From the project root, run:
python -m streamlit run app.py
Streamlit should start a local server and normally open the application
automatically in the default web browser.
The terminal will display a local address similar to:
Local URL: http://localhost:8501
Open the displayed Local URL in a browser if it does not open
automatically.

Step 10 --- Use the Application
Enter the required applicant details:
- Applicant Name
- Customer/Application ID
- Age
- Annual Income
- Home Ownership
- Employment Length
- Credit History Length
- Previous Default Status
- Loan Intent
- Loan Grade
- Loan Amount
- Interest Rate
- Loan Percent of Income
Then click:
Run Credit Risk Assessment
The application will display:
- Risk Category
- Default Probability
- Risk Score
- Applicant information
- Top SHAP factors influencing the prediction
Step 11 --- Stop the Application
Return to the terminal and press:
Ctrl + C
To deactivate the virtual environment:
deactivate

7. One-Page Windows Run Summary
For a new Windows computer, the main commands are:
cd "C:\path\to\credit risk scoring model"

python --version

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python -m streamlit run app.py
After the first setup, the receiver normally only needs:
cd "C:\path\to\credit risk scoring model"

.\.venv\Scripts\Activate.ps1

python -m streamlit run app.py

8. If PowerShell Blocks Virtual Environment Activation
On some Windows systems, PowerShell may display an execution-policy
error when running:
.\.venv\Scripts\Activate.ps1
A temporary solution for the current PowerShell session is:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Then run:
.\.venv\Scripts\Activate.ps1
Alternatively, use Command Prompt:
.venv\Scripts\activate.bat

9. Running the Jupyter Notebook
The Streamlit application can use the already-trained .pkl model, so
retraining is not required just to test the app.
To inspect or reproduce the complete machine-learning workflow, open:
notebooks/credit_risk_model.ipynb
Select the Python interpreter/kernel from the newly created .venv.
The notebook covers:
Data Loading
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Preprocessing
    ↓
Train/Test Split
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Hyperparameter Tuning
    ↓
Risk Scoring
    ↓
SHAP Explainability
    ↓
Save Final Model
Running the notebook is necessary only when reviewing/reproducing the
training workflow or regenerating the model.

10. Dataset
The model uses the following features:
  Feature                        Description
  person_age                   Applicant age
  person_income                Applicant annual income
  person_home_ownership        RENT, OWN, MORTGAGE, OTHER
  person_emp_length            Employment length in years
  loan_intent                  Purpose of requested loan
  loan_grade                   Loan grade A-G
  loan_amnt                    Requested loan amount
  loan_int_rate                Loan interest rate
  loan_percent_income          Loan amount relative to income
  cb_person_default_on_file    Previous default indicator
  cb_person_cred_hist_length   Length of credit history
Target Variable
The target used during model training is:
loan_status
where:
0 = Non-default
1 = Default
loan_status is not provided to the model as an applicant input during
prediction.

11. Models Evaluated
The following classifiers were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Evaluation included:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- False-negative analysis
- Train/test comparison


12. Final Model
Random Forest was selected for tuning based on the overall balance of
predictive performance, explainability, and project suitability.
GridSearchCV used 3-fold cross-validation with F1-score.
Best parameters:
{
    "classifier__max_depth": None,
    "classifier__min_samples_split": 2,
    "classifier__n_estimators": 100
}
Best cross-validation F1-score:
0.8108
Final tuned Random Forest test performance:
  Metric         Score
  Accuracy      0.9274
  Precision     0.9078
  Recall        0.7433
  F1-score      0.8174
  ROC-AUC       0.9289
Confusion Matrix:
TN = 4959
FP = 107
FN = 364
TP = 1054


13. Risk Scoring Logic
The Random Forest does not directly predict Low, Medium, or High.
First, it calculates the probability of default:
probability = model.predict_proba(applicant)[0][1]
Then:
risk_score = probability * 100
Example:
Probability = 0.07
Risk Probability = 7%
Risk Score = 7 / 100
Risk thresholds:
  Default Probability    Category
  <= 0.04              Low
  > 0.04 and <= 0.32   Medium
  > 0.32               High
These thresholds were derived from the predicted-probability
distribution used during project development.
Observed percentiles were:
25th percentile = 0.04
50th percentile = 0.13
75th percentile = 0.32


14. Explainable AI --- SHAP
SHAP is used to explain individual predictions.
The application displays the most influential factors for the current
applicant.
- 🔴 indicates a factor increasing predicted default risk.
- 🟢 indicates a factor decreasing predicted default risk.
SHAP explains the model prediction; it does not determine the
Low/Medium/High threshold category.


15. Application Flow
Applicant Information
        ↓
Preprocessing Pipeline
        ↓
Tuned Random Forest
        ↓
Default Probability
        ↓
Risk Score (0-100)
        ↓
Low / Medium / High Category
        ↓
SHAP Explanation


16. Assessment History
The current application stores assessment history using Streamlit:
st.session_state
This means history is temporary for the running application session.
If the application/server is completely restarted, the previous
assessment history may be lost.
Persistent history can be implemented later using SQLite/PostgreSQL or
another database.

17. Troubleshooting
python is not recognized
Python is either not installed or not available in the system PATH.
Install Python 3 and make sure the option to add Python to PATH is
enabled.
streamlit is not recognized
Make sure the virtual environment is activated and use:
python -m streamlit run app.py
instead of relying only on:
streamlit run app.py
ModuleNotFoundError
Install the dependencies again:
python -m pip install -r requirements.txt
Model file not found
Check that:
models/credit_risk_model.pkl
exists relative to app.py.
PowerShell cannot activate .venv
Use:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
then:
.\.venv\Scripts\Activate.ps1
or use Command Prompt:
.venv\Scripts\activate.bat
App opens but prediction fails
Confirm that:
- the correct trained .pkl model is included,
- package versions are compatible,
- the project folder structure has not been changed,
- all dependencies from requirements.txt were installed.


18. Limitations
- Model performance depends on training-data quality.
- Historical data may not perfectly represent future applicants.
- Risk thresholds may need recalibration for another dataset or
  institution.
- False negatives remain important in credit-risk assessment.
- Model performance may change because of data drift.
- Dedicated fairness and bias analysis is required before real-world
  lending use.
- SHAP explains model behavior but does not establish causation.
- The Streamlit interface is a prototype and does not provide
  production-grade security.
- Current assessment history is session-based.


19. Future Improvements
Possible improvements include:
- Persistent SQLite/PostgreSQL history
- Authentication
- Role-based access control
- Audit logging
- Exportable risk reports
- Data/model drift monitoring
- Automated retraining
- Fairness and bias analysis
- Business-cost-based threshold calibration
- Production-grade security and database architecture


20. Conclusion
This project demonstrates an end-to-end credit-risk machine-learning
workflow.
The final system combines:
Machine Learning
        +
Risk Scoring
        +
Explainable AI
        +
Streamlit User Interface
The tuned Random Forest estimates default probability, the risk-scoring
layer converts the probability into Low/Medium/High risk, and SHAP
explains the factors influencing individual predictions.
For simply testing the completed application on another computer, the
receiver does not need to retrain the model. After installing the
dependencies, the included credit_risk_model.pkl can be loaded
directly by app.py.