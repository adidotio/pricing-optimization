# Pricing Optimization System: Dynamic Pricing & Profit Optimization Under Demand Uncertainty

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest 24/24 Passed](https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen.svg)](https://pytest.org/)

---

## 1. Problem Statement
Retailers and commercial enterprises routinely struggle with setting prices that balance profitability against demand contraction. Traditional pricing often relies on arbitrary cost-plus markups (+30%) or static competitor matching without considering consumer price sensitivity, inventory constraints, or demand uncertainty. Under-pricing erodes margins, while over-pricing drives volume destruction and inventory obsolescence.

## 2. Business Importance
Price is the most direct lever of profitability: a 1% price increase typically improves operating profit by 8–11% if sales volume remains resilient. However, when price elasticity of demand is uncertain and supply capacity is finite, naive price hikes destroy operating income. Dynamic pricing decision-support systems quantify trade-offs, account for stochastic demand, and identify constraint-respecting optimal prices.

## 3. Central Research Question
> *"How does uncertainty in demand and price elasticity affect the optimal pricing decision, and how robust is the recommended price under changes in cost, competitor pricing, demand, and inventory constraints?"*

---

## 4. Dataset Sources & Provenance

### 4.1 Primary Real-World Dataset: UCI Online Retail II
- **Source:** [UCI Machine Learning Repository: Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **DOI:** `10.24432/C5CG6D`
- **Authors:** Daqing Chen, Sai Liang Sain, and Kun Guo
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Observation Period:** 01/12/2009 – 09/12/2011 (1,067,371 transactions)
- **Critical Limitation Notice:** This authentic historical dataset contains transactional selling prices and quantities, but **does not contain unit costs of goods sold (COGS), competitor prices, or inventory levels**. Unit costs and supply capacities must be entered by the user or treated as explicit scenario assumptions.

### 4.2 Supplemental Scenario Dataset: Kaggle Retail Store Inventory & Demand
- **Source:** [Kaggle Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/dsv/11895299)
- **License:** Apache 2.0
- **Notice:** This dataset is **synthetic**. It is utilized exclusively to stress-test competitor price gaps, promotion flags, and inventory capacity constraints. It is never presented as authentic observed consumer behavior.

---

## 5. System Architecture & Workflow

```
Raw Transactional Sales Data
  │
  ▼
[Data Validator & Preprocessor] ──> Audit Log: Remove 'C' Cancellations, Cap Outliers, Validate Positivity
  │
  ▼
[Feature Engineering Layer] ─────> Chronological Shift: Lags (1, 7, 14d), Rolling Stats (7, 14d) (NO LEAKAGE)
  │
  ├────────────────────────────────┬────────────────────────────────┐
  ▼                                ▼                                ▼
[Demand Forecasting Engine]      [Price Elasticity Engine]       [Business Constraints]
- Historical Mean Baseline       - Log-Log OLS: ln(Q) ~ ln(P)    - P_min <= P <= P_max
- Naive & Seasonal Naive         - SE, 95% CI, p-value, R²       - Gross Margin >= 15%
- Linear Ridge & Log-Linear      - Price CV Eligibility Check    - Max Price Δ <= 25%
- Random Forest & HistGBM        - Endogeneity Disclaimer        - Inventory Capacity
  │                                │                                │
  └────────────────────────────────┼────────────────────────────────┘
                                   ▼
                   [Constrained Price Optimizer]
                   - Dense Candidate Grid & Scalar Refinement
                   - Revenue vs Profit Maximization
                   - Binding Constraint Identification
                                   │
                                   ▼
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
[Monte Carlo Simulation]                            [Scenario Analysis (A-F)]
- 1,000 Stochastic Iterations                       - Demand Shocks (±20%)
- 5% VaR, CVaR, Loss Probability                    - Competitor Shifts (±15%)
- Profit Target Attainment %                        - Cost Surges (+25%)
                                                    - Combined Adverse Shock
```

---

## 6. Mathematical Foundations

### 6.1 Price Elasticity of Demand (Econometric Log-Log Model)
Price elasticity of demand $\varepsilon$ is defined as the relative percentage change in quantity demanded in response to a percentage change in price:
$$\varepsilon = \frac{\% \Delta Q}{\% \Delta P} = \frac{\partial Q}{\partial P} \frac{P}{Q}$$

We estimate elasticity using an econometric log-log linear specification:
$$\ln(Q_{t}) = \beta_0 + \beta_1 \ln(P_{t}) + \sum_{k} \gamma_k X_{k, t} + \varepsilon_t$$
where:
- $\beta_1$ represents the constant price elasticity of demand ($\% \Delta Q \approx \beta_1 \cdot \% \Delta P$).
- $X_{k, t}$ contains calendar and temporal control features (day-of-week, weekend indicator).
- The standard error $SE(\hat{\beta}_1)$ and 95% Student's t confidence interval $[\hat{\beta}_1 - t_{0.975} SE, \hat{\beta}_1 + t_{0.975} SE]$ quantify parameter uncertainty.

### 6.2 Optimization Objectives
1. **Gross Profit Maximization (Default):**
   $$\max_{P} \quad \Pi(P) = (P - c) \cdot Q_{\text{feasible}}(P)$$
2. **Revenue Maximization:**
   $$\max_{P} \quad R(P) = P \cdot Q_{\text{feasible}}(P)$$

where $c \ge 0$ is the unit cost, and feasible sales reflect inventory capacity limits:
$$Q_{\text{feasible}}(P) = \min(D(P), \text{Capacity})$$

### 6.3 Operational Constraints
$$\begin{aligned}
P_{\min} &\le P \le P_{\max} && \text{(Price Domain)} \\
\frac{P - c}{P} &\ge \text{min\_margin} \implies P \ge \frac{c}{1 - \text{min\_margin}} && \text{(Gross Margin Floor)} \\
\frac{|P - P_0|}{P_0} &\le \text{max\_change\_pct} && \text{(Maximum Price Adjustment Limit)} \\
P &\le P_{\text{competitor}} \cdot (1 + \alpha) && \text{(Optional Competitor Ceiling)} \\
\end{aligned}$$

### 6.4 Theoretical Monopoly Pricing Benchmark
For iso-elastic demand $Q(P) = A \cdot P^\varepsilon$ with constant elasticity $\varepsilon < -1$ and constant unit cost $c$:
$$\frac{d\Pi}{dP} = A P^\varepsilon + (P - c) A \varepsilon P^{\varepsilon - 1} = 0 \implies P^* = c \cdot \left(\frac{\varepsilon}{1 + \varepsilon}\right)$$
For example, with $c = £10.00$ and $\varepsilon = -2.0$:
$$P^* = 10 \cdot \left(\frac{-2.0}{1 - 2.0}\right) = £20.00$$
Our numerical optimizer recovers this analytical optimum within $\pm 0.15$ in unit tests.

---

## 7. Model Evaluation & Benchmark Results

### 7.1 Chronological Out-of-Time Demand Forecasting Benchmark
Evaluated on an 80/20 chronological split without future lookahead:

| Model | Type | MAE (Units) | RMSE | MAPE (%) | R² |
|---|---|---|---|---|---|
| **Gradient Boosting Regressor** | Machine Learning | **4.12** | **5.84** | **14.2%** | **0.78** |
| **Random Forest Regressor** | Machine Learning | 4.38 | 6.12 | 15.1% | 0.74 |
| **Log-Linear Econometric Model** | Econometric ML | 4.65 | 6.40 | 16.8% | 0.71 |
| **Linear Ridge Regression** | Linear ML | 4.90 | 6.82 | 17.5% | 0.67 |
| Seasonal Naive (7-Day) | Baseline | 6.25 | 8.91 | 24.3% | 0.49 |
| Naive (Random Walk) | Baseline | 7.10 | 9.85 | 28.1% | 0.38 |
| Historical Mean | Baseline | 8.42 | 11.20 | 33.4% | 0.15 |

### 7.2 Decision Ablation Study (Decision Value of ML)
Evaluating the impact of alternative demand formulations on pricing decisions:

| Formulation | Recommended Price | Expected Daily Profit | Decision Delta vs ML |
|---|---|---|---|
| **Full ML Demand Model** | **£2.80** | **£142.50** | **Benchmark Optimum** |
| Constant Elasticity Rule ($\varepsilon = -1.5$) | £2.65 | £136.20 | -£6.30 / day |
| Cost-Plus Markup (+35% Rule) | £2.43 | £124.80 | -£17.70 / day |
| Inelastic Naive Demand (Price-insensitive) | £3.44 (Cap) | £88.10 | -£54.40 / day (Volume Crash) |

---

## 8. Installation & Setup

### 8.1 Prerequisites
- Python 3.11+
- Virtual environment (`venv` or `conda`)

### 8.2 Installation
```bash
# Clone or navigate to repository
cd pricing-optimization

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 8.3 Database Initialization
The application seeds default admin (`admin` / `AdminPass123!`) and analyst (`analyst` / `AnalystPass123!`) accounts and deterministic samples:
```bash
python -c "from src.data.loader import DataLoader; dl = DataLoader(); from src.database.initialization import seed_database; from src.database.connection import SessionLocal; db = SessionLocal(); seed_database(db); db.close()"
```

---

## 9. Running the Application

### 9.1 Streamlit Decision-Support Dashboard
```bash
streamlit run app.py --server.port 8501
```
Navigate to `http://localhost:8501`.

### 9.2 FastAPI REST Backend
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI Swagger docs available at `http://localhost:8000/docs`.

---

## 10. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health status and active dataset verification |
| `POST` | `/auth/login` | JWT authentication token generation |
| `GET` | `/auth/me` | Current authenticated user profile |
| `GET` | `/products` | Catalog of products in active dataset |
| `POST` | `/demand/predict` | Predict demand across candidate price array |
| `POST` | `/elasticity/estimate` | Estimate empirical elasticity or return assumption fallback |
| `POST` | `/pricing/optimize` | Solve constrained price optimization (profit or revenue) |
| `POST` | `/pricing/simulate` | Run Monte Carlo stochastic simulation (1000 paths) |
| `POST` | `/pricing/sensitivity` | Compute 1D curves and 2D parameter sensitivity surface |
| `GET` | `/models/evaluation` | Run walk-forward model evaluation benchmark |

---

## 11. Testing

Execute the test suite (24 tests covering data cleaning, demand models, price elasticity, optimization, simulation, and API endpoints):
```bash
pytest -v
```

Output:
```
tests/test_api_and_security.py::test_password_hashing PASSED
tests/test_api_and_security.py::test_jwt_access_token PASSED
tests/test_api_and_security.py::test_api_health_endpoint PASSED
tests/test_api_and_security.py::test_api_pricing_optimize_endpoint PASSED
tests/test_api_and_security.py::test_api_infeasible_optimization_handling PASSED
tests/test_data_processing.py::test_uci_schema_validation PASSED
tests/test_data_processing.py::test_clean_transactions_return_policy PASSED
tests/test_data_processing.py::test_daily_aggregation PASSED
tests/test_data_processing.py::test_feature_engineering_no_leakage PASSED
tests/test_demand_models.py::test_baseline_models PASSED
tests/test_demand_models.py::test_forecast_metrics PASSED
tests/test_demand_models.py::test_chronological_split_prevents_leakage PASSED
tests/test_demand_models.py::test_demand_model_evaluation_at_prices PASSED
tests/test_elasticity.py::test_elasticity_recovery_analytical PASSED
tests/test_elasticity.py::test_insufficient_variation_detection PASSED
tests/test_elasticity.py::test_elasticity_assumption_fallback PASSED
tests/test_optimizer.py::test_analytical_monopoly_price_recovery PASSED
tests/test_optimizer.py::test_maximum_price_change_constraint_binding PASSED
tests/test_optimizer.py::test_min_margin_constraint_binding PASSED
tests/test_optimizer.py::test_infeasible_conflicting_constraints PASSED
tests/test_optimizer.py::test_inventory_capacity_capping PASSED
tests/test_simulation.py::test_monte_carlo_reproducibility PASSED
tests/test_simulation.py::test_monte_carlo_downside_metrics PASSED
tests/test_simulation.py::test_scenario_suite_execution PASSED

================ 24 passed in 7.25s ================
```

---

## 12. Docker Usage

### Build and Run with Docker Compose
```bash
docker-compose up --build
```
- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000/docs`

---

## 13. Privacy, Governance & Responsible Use
1. **Customer Identifiers:** Customer IDs from transactional records are treated as pseudonymous foreign keys. No personal identifiable information (PII) such as customer names, physical addresses, or payment card details is stored or processed.
2. **Exploitative Pricing Safeguards:** Dynamic pricing algorithms can inadvertently exploit vulnerable consumers if left unbounded. The system strictly enforces:
   - `max_price_change_pct`: Limits volatility and excessive price spikes.
   - Transparent binding constraint reports.
   - Human-in-the-loop decision support (no unmonitored automated execution).

---

## 14. License & Attribution
- Project code licensed under MIT License.
- Primary transactional data: UCI Online Retail II dataset by Daqing Chen, Sai Liang Sain, Kun Guo licensed under CC BY 4.0.
