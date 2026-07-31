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
.links a{color:#60a5fa;text-decoration:none;font-size:13px;background:#1a2333;border:1px solid #2a3a5a;padding:6px 12px;border-radius:6px}
.links a:hover{background:#223049}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
.card{background:#161a23;border:1px solid #2a2f3f;border-radius:12px;padding:20px}
.card h2{font-size:16px;margin-bottom:16px;color:#fff;border-bottom:1px solid #2a2f3f;padding-bottom:10px}
.form-group{margin-bottom:14px}
label{display:block;font-size:12px;color:#9aa0b2;margin-bottom:6px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px}
input,select{width:100%;background:#0f1115;border:1px solid #2a2f3f;color:#e6e6e6;padding:10px 12px;border-radius:8px;font-size:14px}
input:focus,select:focus{outline:none;border-color:#60a5fa;box-shadow:0 0 0 2px rgba(96,165,250,0.2)}
.hint{font-size:11px;color:#6b7280;margin-top:4px}
button{width:100%;background:#3b82f6;color:#fff;border:none;padding:14px;border-radius:8px;font-size:15px;font-weight:600;cursor:pointer;margin-top:10px}
button:hover{background:#2563eb}
.result{margin-top:20px;padding:16px;border-radius:10px;display:none}
.result.show{display:block}
.result.low{background:#102a1a;border:1px solid #1f5a30;color:#4ade80}
.result.high{background:#2a1515;border:1px solid #5a2323;color:#f87171}
.result.error{background:#2a2215;border:1px solid #5a4a23;color:#fbbf24}
.result-title{font-weight:700;font-size:18px;margin-bottom:6px}
.result-prob{font-size:13px;opacity:0.9}
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
      <a href="/docs" target="_blank">📄 Swagger /docs</a>
      <a href="/health" target="_blank">💚 /health</a>
      <a href="https://github.com/RohanDey11/credit-risk-api" target="_blank">⭐ GitHub</a>
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Applicant Details</h2>
      <form id="riskForm" novalidate>
        <div class="form-group"><label>Age</label><input type="number" name="person_age" value="32" min="18" max="100" required><div class="hint">18-100, 15 returns 422 from server</div></div>
        <div class="form-group"><label>Income ($)</label><input type="number" name="person_income" value="59000" min="1" required></div>
        <div class="form-group"><label>Home Ownership</label><select name="person_home_ownership" required><option value="RENT" selected>RENT - 2x default risk</option><option value="MORTGAGE">MORTGAGE</option><option value="OWN">OWN</option><option value="OTHER">OTHER</option></select></div>
        <div class="form-group"><label>Employment Length</label><input type="number" name="person_emp_length" value="3" step="0.5" min="0" max="100" required></div>
        <div class="form-group"><label>Loan Intent</label><select name="loan_intent" required><option value="MEDICAL" selected>MEDICAL</option><option value="EDUCATION">EDUCATION</option><option value="DEBTCONSOLIDATION">DEBT CONSOLIDATION</option><option value="HOMEIMPROVEMENT">HOME IMPROVEMENT</option><option value="PERSONAL">PERSONAL</option><option value="VENTURE">VENTURE</option></select></div>
        <div class="form-group"><label>Loan Grade</label><select name="loan_grade" required><option value="A">A</option><option value="B">B</option><option value="C" selected>C</option><option value="D">D</option><option value="E">E</option><option value="F">F</option><option value="G">G</option></select></div>
        <div class="form-group"><label>Loan Amount ($)</label><input type="number" name="loan_amnt" value="5500" min="1" max="50000" required></div>
        <div class="form-group"><label>Interest Rate (%)</label><input type="number" name="loan_int_rate" value="12.87" step="0.01" min="5" max="30" required></div>
        <div class="form-group"><label>Percent Income (0-1)</label><input type="number" name="loan_percent_income" value="0.09" step="0.01" min="0" max="1" required><div class="hint">Top feature 0.26 - >0.30 high risk</div></div>
        <div class="form-group"><label>Default on File</label><select name="cb_person_default_on_file" required><option value="N" selected>N - No</option><option value="Y">Y - Yes</option></select></div>
        <div class="form-group"><label>Credit History Length</label><input type="number" name="cb_person_cred_hist_length" value="3" min="0" max="30" required></div>
        <button type="submit" id="submitBtn">▶ Predict Risk - 12ms</button>
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
        <div class="metric"><span>Latency</span><span>~12ms</span></div>
        <div class="insight"><div class="insight-title">Business Insight</div><div class="insight-text"><b>loan_percent_income 0.26</b> strongest. >30% high risk.<br><br><b>RENT 2x default</b> vs OWN.<br><br>Recall 76% intentional.</div></div>
      </div>
      <div class="card">
        <h2>Validation Edge</h2>
        <p style="font-size:13px;color:#9aa0b2;margin-bottom:12px">Most freelance code crashes on bad input. This returns 422 from Pydantic V2 before model.</p>
        <div style="background:#0f1115;border:1px solid #2a2f3f;border-radius:8px;padding:12px;font-family:monospace;font-size:12px;color:#9aa0b2">Try Age=15 → 422 Unprocessable Entity<br>(server-side, not browser tooltip)</div>
      </div>
    </div>
  </div>
  <div class="footer">Rohan Dey — MSc Data Science | Ex AI Research Intern @ TCG CREST | BS Data Science, IIT Madras<br>FastAPI + Pydantic V2 + Docker + Render + UptimeRobot + pytest</div>
</div>
<script>
document.getElementById('riskForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('submitBtn');
  const resultDiv = document.getElementById('result');
  btn.textContent = '⏳ Predicting...'; btn.disabled = true;
  const formData = new FormData(e.target);
  const data = {};
  for (let [k,v] of formData.entries()) {
    if (['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length'].includes(k)) data[k]=parseFloat(v); else data[k]=v;
  }
  try {
    const res = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const json = await res.json();
    if (!res.ok) {
      resultDiv.className='result error show';
      resultDiv.innerHTML=`<div class="result-title">⚠️ Validation ${res.status} - Pydantic V2</div><pre style="white-space:pre-wrap;font-size:11px">${JSON.stringify(json,null,2)}</pre>`;
    } else {
      const isHigh = json.prediction===1 || (json.probability&&json.probability>0.5);
      resultDiv.className=isHigh?'result high show':'result low show';
      const label=json.risk||(isHigh?'High Risk - Likely Default':'Low Risk - Likely Repay');
      const prob=json.probability!==undefined?`Prob: ${(json.probability*100).toFixed(1)}%`:'';
      resultDiv.innerHTML=`<div class="result-title">${isHigh?'🔴':'🟢'} ${label}</div><div class="result-prob">${prob} | ${json.latency_ms||''}ms</div>`;
    }
  } catch(err){resultDiv.className='result error show';resultDiv.innerHTML=`<div class="result-title">❌ Error</div>${err.message}`;}
  finally{btn.textContent='▶ Predict Risk - 12ms';btn.disabled=false;}
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
