# Credit-Risk API — 0.9214 ROC-AUC (Live)

A production-ready credit risk classifier, deployed from notebook to a live, Dockerized FastAPI service with input validation, health monitoring, and Swagger docs.

**Live Demo:** https://credit-risk-api-sdeq.onrender.com/
**API Docs:** https://credit-risk-api-sdeq.onrender.com/docs
**Health Check:** https://credit-risk-api-sdeq.onrender.com/health

> Note: hosted on Render's free tier — the service sleeps after 15 min of inactivity. First request after sleep can take ~20s to spin back up; subsequent requests are fast.

---

## What This Is

A tuned Random Forest credit risk model (trained on 32,576 loan applications, 22 features) taken from a research notebook and productionized into a real API — the kind of "notebook-to-deployment" work most portfolios skip.

| Metric       | Value                               |
| ------------ | ----------------------------------- |
| ROC-AUC      | 0.9214                              |
| Recall       | 0.76                                |
| Model        | Random Forest (max_depth=10, tuned) |
| Features     | 22                                  |
| Dataset size | 32,576 rows                         |

## Business Insights (from EDA)

- **`loan_percent_income`** is the single strongest predictor (0.26 feature importance) — borrowers spending over ~30% of income on loan payments are disproportionately high-risk.
- **Renters default at ~2x the rate of homeowners**, holding other factors constant.
- Recall was deliberately tuned to 76% — in credit risk, missing an actual defaulter is far costlier than a false alarm, so the model favors catching risk over minimizing false positives.

## Before / After

**Before:** Notebook model — 0.9214 ROC-AUC, but no input validation, no error handling, crashes on malformed input, unusable outside a Jupyter session.

**After:** FastAPI + Pydantic V2 validation, Dockerized, `/health` monitoring endpoint, Swagger UI, deployed live on Render, ~12ms inference latency, malformed input returns a proper `422` instead of a crash.

## Architecture

```
Client → FastAPI (Pydantic V2 validation) → Preprocessing (22 features) → Random Forest (max_depth=10) → JSON response
```

## API Endpoints

| Endpoint   | Method | Description                                                        |
| ---------- | ------ | ------------------------------------------------------------------ |
| `/`        | GET    | Interactive form — submit applicant details, get a live prediction |
| `/predict` | POST   | JSON API endpoint — returns risk classification                    |
| `/health`  | GET    | Returns model status, ROC-AUC, recall, uptime                      |
| `/docs`    | GET    | Swagger UI — interactive API documentation                         |

### Example Request

```bash
curl -X POST https://credit-risk-api-sdeq.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 25,
    "person_income": 55000,
    "person_home_ownership": "RENT",
    "loan_intent": "MEDICAL",
    "loan_grade": "C",
    "loan_amnt": 5500
  }'
```

## Validation Example

Invalid input is rejected before it ever reaches the model:

```bash
curl -X POST https://credit-risk-api-sdeq.onrender.com/predict \
  -d '{"person_age": 15, ...}'
# → 422 Unprocessable Entity
```

## Tech Stack

- **FastAPI** — API framework
- **Pydantic V2** — request/response validation
- **scikit-learn** — model (Random Forest)
- **Docker** — containerized deployment
- **Render** — hosting
- **UptimeRobot** — uptime monitoring
- **pytest** — test suite (health check, valid prediction, invalid input handling)

## Running Locally

```bash
git clone https://github.com/RohanDey11/credit-risk-api.git
cd credit-risk-api
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then visit `http://localhost:8000`.

## Running with Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## Tests

```bash
pytest tests/ -v
```

Covers: health check, valid prediction flow, and rejection of invalid input (e.g., out-of-range age).

---

**Author:** Rohan Dey | Data Science Engineer — MSc Data Science | Ex AI Research Intern @ TCG CREST | BS Data Science, IIT Madras
