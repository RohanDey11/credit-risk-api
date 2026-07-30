from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
def test_predict_valid():
    payload = {"person_age":32,"person_income":59000,"person_home_ownership":"RENT","person_emp_length":3.0,"loan_intent":"MEDICAL","loan_grade":"C","loan_amnt":5500,"loan_int_rate":12.87,"loan_percent_income":0.09,"cb_person_default_on_file":"N","cb_person_cred_hist_length":3}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
def test_invalid_age_422():
    payload = {"person_age":15,"person_income":59000,"person_home_ownership":"RENT","person_emp_length":3.0,"loan_intent":"MEDICAL","loan_grade":"C","loan_amnt":5500,"loan_int_rate":12.87,"loan_percent_income":0.09,"cb_person_default_on_file":"N","cb_person_cred_hist_length":3}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
