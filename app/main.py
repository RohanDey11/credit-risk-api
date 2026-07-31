from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import json
import time
from pathlib import Path

from app.schemas import CreditRiskInput
from app.preprocess import preprocess_input

app = FastAPI(title="Credit-Risk API - 0.9214 ROC-AUC", version="1.0.0")

MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"
COLUMNS_PATH = Path(__file__).parent / "models" / "columns.json"

model = None
columns = None
model_loaded = False

try:
    if MODEL_PATH.exists() and COLUMNS_PATH.exists():
        model = joblib.load(MODEL_PATH)
        with open(COLUMNS_PATH, 'r') as f:
            columns = json.load(f)
        model_loaded = True
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"Model files not found")
except Exception as e:
    print(f"Error loading model: {e}")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Credit-Risk API - 0.9214 ROC-AUC Live</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f1115;color:#e6e6e6;line-height:1.5;padding:20px}
.container{max-width:900px;margin:0 auto}
.header{background:#161a23;border:1px solid #2a2f3f;border-radius:12px;padding:24px;margin-bottom:20px}
.badge{display:inline-block;background:#1e3a2a;color:#4ade80;border:1px solid #2d5a3d;padding:4px 10px;border-radius:20px;font-size:12px;font-weight:600;margin-bottom:12px}
h1{font-size:24px;font-weight:700;margin-bottom:8px;color:#fff}
.sub{color:#9aa0b2;font-size:14px;margin-bottom:16px}
.links{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
.links a{color:#c9d1e0;text-decoration:none;font-size:13px;background:#1a2333;border:1px solid #2a3a5a;padding:7px 12px;border-radius:6px;display:inline-flex;align-items:center;gap:6px;font-weight:500}
.links a:hover{background:#223049;color:#fff;border-color:#3a4a6a}
.links a svg{width:14px;height:14px;opacity:0.8}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
.card{background:#161a23;border:1px solid #2a2f3f;border-radius:12px;padding:20px}
.card h2{font-size:16px;margin-bottom:16px;color:#fff;border-bottom:1px solid #2a2f3f;padding-bottom:10px}
.form-group{margin-bottom:14px}
label{display:block;font-size:12px;color:#9aa0b2;margin-bottom:6px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px}
input,select{width:100%;background:#0f1115;border:1px solid #2a2f3f;color:#e6e6e6;padding:10px 12px;border-radius:8px;font-size:14px;transition:border-color 0.2s}
input:focus,select:focus{outline:none;border-color:#60a5fa;box-shadow:0 0 0 2px rgba(96,165,250,0.2)}
input[readonly]{background:#1a1f2e;color:#a0aec0;cursor:default;border-color:#2a344f}
/* Hide spinner for readonly to avoid Firefox quirk */
input[readonly]::-webkit-outer-spin-button,
input[readonly]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
input[readonly]{-moz-appearance:textfield}
.hint{font-size:11px;color:#6b7280;margin-top:4px}
button{width:100%;background:#3b82f6;color:#fff;border:none;padding:14px;border-radius:8px;font-size:15px;font-weight:600;cursor:pointer;margin-top:10px;transition:background 0.2s}
button:hover{background:#2563eb}
button:disabled{background:#2a344f;color:#6b7280;cursor:not-allowed}
.cold-note{font-size:11px;color:#6b7280;text-align:center;margin-top:8px;line-height:1.3}
.result{margin-top:20px;padding:16px;border-radius:10px;display:none}
.result.show{display:block}
.result.low{background:#102a1a;border:1px solid #1f5a30;color:#4ade80}
.result.high{background:#2a1515;border:1px solid #5a2323;color:#f87171}
.result.error{background:#2a2215;border:1px solid #5a4a23;color:#fbbf24}
.result-title{font-weight:700;font-size:16px;margin-bottom:6px}
.result-friendly{font-size:14px;font-weight:600;margin-bottom:6px;line-height:1.4}
.result-friendly-list{margin-bottom:10px}
.result-prob{font-size:13px;opacity:0.9;margin-top:8px}
.result pre{white-space:pre-wrap;font-size:11px;background:rgba(0,0,0,0.3);padding:10px;border-radius:6px;margin-top:10px;overflow-x:auto}
.insight{background:#1a1f2e;border:1px solid #2a344f;border-radius:8px;padding:12px;margin-top:16px}
.insight-title{font-size:12px;font-weight:600;color:#60a5fa;margin-bottom:6px;text-transform:uppercase}
.insight-text{font-size:13px;color:#9aa0b2;line-height:1.4}
.footer{text-align:center;margin-top:24px;color:#6b7280;font-size:12px}
.metric{display:flex;justify-content:space-between;background:#0f1115;border:1px solid #2a2f3f;padding:8px 12px;border-radius:6px;margin-bottom:8px;font-size:13px}
.metric span:last-child{font-weight:600;color:#fff}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="badge">● LIVE - ROC-AUC 0.9214 | Recall 76%</div>
    <h1>Credit-Risk API — Live Demo</h1>
    <p class="sub">Tuned Random Forest max_depth=10 — 32,576 loans, 22 features — Dockerized FastAPI</p>
    <div class="links">
      <a href="/docs" target="_blank">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
        Swagger Docs
      </a>
      <a href="/health" target="_blank">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/><circle cx="12" cy="12" r="10" opacity="0.3"/></svg>
        Health
      </a>
      <a href="https://github.com/RohanDey11/credit-risk-api" target="_blank">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
        GitHub
      </a>
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Applicant Details</h2>
      <form id="riskForm" novalidate>
        <div class="form-group">
          <label for="person_age">Age</label>
          <input type="number" id="person_age" name="person_age" value="32" min="0" max="120" step="1" inputmode="numeric" required>
          <div class="hint">Must be 18 or older</div>
        </div>
        <div class="form-group">
          <label for="person_income">Income ($)</label>
          <input type="number" id="person_income" name="person_income" value="59000" min="1" max="10000000" step="1000" inputmode="numeric" required>
        </div>
        <div class="form-group">
          <label for="person_home_ownership">Home Ownership</label>
          <select id="person_home_ownership" name="person_home_ownership" required>
            <option value="RENT" selected>RENT</option>
            <option value="MORTGAGE">MORTGAGE</option>
            <option value="OWN">OWN</option>
            <option value="OTHER">OTHER</option>
          </select>
          <div class="hint">Renters show ~2x default rate in data</div>
        </div>
        <div class="form-group">
          <label for="person_emp_length">Employment Length (years)</label>
          <input type="number" id="person_emp_length" name="person_emp_length" value="3" min="0" max="100" step="0.5" inputmode="decimal" required>
        </div>
        <div class="form-group">
          <label for="loan_intent">Loan Intent</label>
          <select id="loan_intent" name="loan_intent" required>
            <option value="MEDICAL" selected>MEDICAL</option>
            <option value="EDUCATION">EDUCATION</option>
            <option value="DEBTCONSOLIDATION">DEBT CONSOLIDATION</option>
            <option value="HOMEIMPROVEMENT">HOME IMPROVEMENT</option>
            <option value="PERSONAL">PERSONAL</option>
            <option value="VENTURE">VENTURE</option>
          </select>
        </div>
        <div class="form-group">
          <label for="loan_grade">Loan Grade</label>
          <select id="loan_grade" name="loan_grade" required>
            <option value="A">A - Lowest risk</option>
            <option value="B">B</option>
            <option value="C" selected>C</option>
            <option value="D">D</option>
            <option value="E">E</option>
            <option value="F">F</option>
            <option value="G">G - Highest risk</option>
          </select>
        </div>
        <div class="form-group">
          <label for="loan_amnt">Loan Amount ($)</label>
          <input type="number" id="loan_amnt" name="loan_amnt" value="5500" min="1" max="50000" step="100" inputmode="numeric" required>
        </div>
        <div class="form-group">
          <label for="loan_int_rate">Interest Rate (%)</label>
          <input type="number" id="loan_int_rate" name="loan_int_rate" value="12.87" min="5" max="30" step="0.01" inputmode="decimal" required>
        </div>
        <div class="form-group">
          <label for="loan_percent_income">Percent Income (auto-calculated)</label>
          <input type="number" id="loan_percent_income" name="loan_percent_income" value="0.09" min="0" max="1" step="0.0001" inputmode="decimal" readonly>
          <div class="hint">Auto-calculated as loan / income — strongest predictor (0.26 importance)</div>
        </div>
        <div class="form-group">
          <label for="cb_person_default_on_file">Default on File</label>
          <select id="cb_person_default_on_file" name="cb_person_default_on_file" required>
            <option value="N" selected>N - No</option>
            <option value="Y">Y - Yes, prior default</option>
          </select>
        </div>
        <div class="form-group">
          <label for="cb_person_cred_hist_length">Credit History Length (years)</label>
          <input type="number" id="cb_person_cred_hist_length" name="cb_person_cred_hist_length" value="3" min="0" max="30" step="1" inputmode="numeric" required>
        </div>
        <button type="submit" id="submitBtn">Predict Risk</button>
        <div class="cold-note">First request may take ~20s if server was idle (free tier). Subsequent requests are fast.</div>
      </form>
      <div id="result" class="result"></div>
    </div>
    <div>
      <div class="card" style="margin-bottom:20px">
        <h2>Model Metrics</h2>
        <div class="metric"><span>ROC-AUC</span><span>0.9214</span></div>
        <div class="metric"><span>Recall</span><span>76% tuned</span></div>
        <div class="metric"><span>Model</span><span>RF max_depth 10</span></div>
        <div class="metric"><span>Rows</span><span>32,576</span></div>
        <div class="metric"><span>Latency</span><span>Measured live</span></div>
        <div class="insight"><div class="insight-title">Business Insight</div><div class="insight-text"><b>loan_percent_income 0.26</b> strongest predictor. >30% income on loan = high risk.<br><br><b>RENT ~2x default</b> vs OWN in this dataset, holding other factors.<br><br>Recall 76% is intentional — missing a defaulter costs more than false alarm.</div></div>
      </div>
      <div class="card">
        <h2>Validation Edge</h2>
        <p style="font-size:13px;color:#9aa0b2;margin-bottom:12px">Invalid input is rejected server-side by Pydantic V2 before reaching the model — not just browser checks.</p>
        <div style="background:#0f1115;border:1px solid #2a2f3f;border-radius:8px;padding:12px;font-family:monospace;font-size:12px;color:#9aa0b2">Example: Age below 18 → server returns 422 with clear error<br>Technical detail shown below for transparency</div>
      </div>
    </div>
  </div>
  <div class="footer">Rohan Dey — MSc Data Science | Ex AI Research Intern @ TCG CREST | BS Data Science, IIT Madras<br>FastAPI + Pydantic V2 + Docker + Render + UptimeRobot + pytest</div>
</div>
<script>
// Auto-calculate loan_percent_income = loan_amnt / person_income
function calcPercent() {
  const incomeEl = document.getElementById('person_income');
  const loanEl = document.getElementById('loan_amnt');
  const percentEl = document.getElementById('loan_percent_income');
  const income = parseFloat(incomeEl.value);
  const loan = parseFloat(loanEl.value);
  if (income && income > 0 && loan >= 0) {
    const pct = loan / income;
    percentEl.value = pct.toFixed(4);
  }
}
document.getElementById('person_income').addEventListener('input', calcPercent);
document.getElementById('loan_amnt').addEventListener('input', calcPercent);
calcPercent();

function friendlySingleError(err) {
  const loc = err.loc || [];
  const field = loc[loc.length - 1] || 'input';
  const msg = err.msg || '';
  const fieldNames = {
    'person_age': 'Age',
    'person_income': 'Income',
    'person_home_ownership': 'Home Ownership',
    'person_emp_length': 'Employment Length',
    'loan_intent': 'Loan Intent',
    'loan_grade': 'Loan Grade',
    'loan_amnt': 'Loan Amount',
    'loan_int_rate': 'Interest Rate',
    'loan_percent_income': 'Percent Income',
    'cb_person_default_on_file': 'Default on File',
    'cb_person_cred_hist_length': 'Credit History Length'
  };
  const friendlyField = fieldNames[field] || field;
  if (field === 'person_age' && msg.includes('greater than or equal to 18')) return `Age must be 18 or older`;
  if (field === 'person_age' && msg.includes('less than or equal to')) return `Age must be 100 or younger`;
  if (field === 'person_income' && msg.includes('greater than 0')) return `Income must be greater than 0`;
  if (field === 'person_income' && msg.includes('less than or equal to')) return `Income must be 10,000,000 or less`;
  if (field === 'loan_amnt' && msg.includes('greater than 0')) return `Loan amount must be greater than 0`;
  if (field === 'loan_amnt' && msg.includes('less than or equal to')) return `Loan amount must be 50,000 or less`;
  if (field === 'loan_int_rate' && msg.includes('greater than or equal to 5')) return `Interest rate must be at least 5%`;
  if (field === 'loan_int_rate' && msg.includes('less than or equal to')) return `Interest rate must be 30% or less`;
  if (field === 'loan_percent_income' && msg.includes('less than or equal to 1')) return `Percent income must be 1 or less`;
  return `${friendlyField}: ${msg}`;
}

function friendlyErrors(details) {
  if (!details || !Array.isArray(details) || details.length === 0) return [];
  return details.map(friendlySingleError);
}

document.getElementById('riskForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('submitBtn');
  const resultDiv = document.getElementById('result');
  const originalText = btn.textContent;
  btn.textContent = 'Predicting...';
  btn.disabled = true;
  calcPercent();
  const formData = new FormData(e.target);
  const data = {};
  for (let [k,v] of formData.entries()) {
    if (['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length'].includes(k)) {
      data[k]=parseFloat(v);
    } else {
      data[k]=v;
    }
  }
  try {
    const res = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const json = await res.json();
    if (!res.ok) {
      resultDiv.className='result error show';
      const friendlies = friendlyErrors(json.detail);
      let html = '';
      if (friendlies.length > 0) {
        html += `<div class="result-friendly-list">`;
        friendlies.forEach(f => {
          html += `<div class="result-friendly">❌ ${f}</div>`;
        });
        html += `</div>`;
      } else {
        html += `<div class="result-title">Validation Error (${res.status})</div>`;
      }
      html += `<div style="font-size:12px;opacity:0.8;margin-bottom:6px">Technical detail (Pydantic V2):</div>`;
      html += `<pre>${JSON.stringify(json,null,2)}</pre>`;
      resultDiv.innerHTML = html;
    } else {
      const isHigh = json.prediction===1 || (json.probability&&json.probability>0.5);
      resultDiv.className=isHigh?'result high show':'result low show';
      const label=json.risk||(isHigh?'High Risk - Likely Default':'Low Risk - Likely Repay');
      const prob=json.probability!==undefined?`Probability: ${(json.probability*100).toFixed(1)}%`:'';
      const latency = json.latency_ms ? `${json.latency_ms}ms` : '';
      resultDiv.innerHTML=`<div class="result-title">${isHigh?'High Risk':'Low Risk'}: ${label}</div><div class="result-prob">${prob}${prob && latency ? ' • ' : ''}${latency}</div>`;
    }
  } catch(err){
    resultDiv.className='result error show';
    resultDiv.innerHTML=`<div class="result-title">Connection Error</div><div class="result-friendly">Could not reach server. First request may take ~20s if server was idle.</div><pre>${err.message}</pre>`;
  }
  finally{
    btn.textContent=originalText;
    btn.disabled=false;
  }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE

@app.api_route("/health", methods=["GET", "HEAD", "POST", "OPTIONS"])
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "roc_auc": 0.9214,
        "recall": 0.76,
        "model": "Random Forest max_depth 10",
        "features": 22,
        "rows": 32576
    }

@app.post("/predict")
async def predict(request: CreditRiskInput):
    start = time.time()
    try:
        input_dict = request.model_dump()
        features = preprocess_input(input_dict)
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})
        prob = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else float(model.predict(features)[0])
        pred = int(prob > 0.5)
        latency = int((time.time() - start) * 1000)
        risk_label = "High Risk - Likely Default" if pred == 1 else "Low Risk - Likely Repay"
        return {
            "prediction": pred,
            "risk": risk_label,
            "probability": float(prob),
            "default_probability": float(prob),
            "roc_auc": 0.9214,
            "model": "Random Forest max_depth 10",
            "latency_ms": latency
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})
