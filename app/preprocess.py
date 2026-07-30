import pandas as pd
import json
from pathlib import Path

COLUMNS_PATH = Path(__file__).parent / "models" / "columns.json"

def load_expected_columns():
    if COLUMNS_PATH.exists():
        with open(COLUMNS_PATH, 'r') as f:
            return json.load(f)
    return [
        "person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
        "loan_percent_income", "cb_person_cred_hist_length",
        "person_home_ownership_OTHER", "person_home_ownership_OWN", "person_home_ownership_RENT",
        "loan_intent_EDUCATION", "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL", "loan_intent_PERSONAL", "loan_intent_VENTURE",
        "loan_grade_B", "loan_grade_C", "loan_grade_D", "loan_grade_E", "loan_grade_F", "loan_grade_G",
        "cb_person_default_on_file_Y"
    ]

EXPECTED_COLS = load_expected_columns()

def preprocess_input(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_final = df_encoded.reindex(columns=EXPECTED_COLS, fill_value=0)
    return df_final
