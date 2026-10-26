"""Model benchmarking, ablation experiments, and pricing policy decision evaluation."""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.models.baseline import HistoricalMeanDemandBaseline, NaiveDemandBaseline, SeasonalNaiveBaseline
from src.models.demand_model import (
    LinearRidgeDemandModel, LogLinearDemandModel, 
    RandomForestDemandModel, GradientBoostingDemandModel
)
from src.evaluation.metrics import calculate_forecast_metrics
from src.evaluation.temporal_validation import TemporalValidator
from src.preprocessing.feature_engineering import FeatureEngineer
from src.optimization.objective import PricingObjective
from src.optimization.constraints import PricingConstraints
from src.optimization.optimizer import PriceOptimizer
from src.utils.logger import logger


class ModelBenchmarkExperiment:
    """Executes chronological benchmarking across baseline and ML demand forecasting models."""

    @classmethod
    def run_benchmark(
        cls, 
        feat_df: pd.DataFrame, 
        product_id: str,
        test_size: float = 0.20
    ) -> Dict[str, Any]:
        """Trains and evaluates baselines and ML models on a chronological split for a single product."""
        prod_data = feat_df[feat_df["product_id"] == str(product_id)].sort_values("date").copy()
        if len(prod_data) < 15:
            return {
                "product_id": product_id,
                "error": f"Insufficient data ({len(prod_data)} rows) for reliable temporal evaluation."
            }

        train_df, test_df = TemporalValidator.chronological_split(prod_data, test_size=test_size)
        y_train = train_df["quantity"]
        y_test = test_df["quantity"]
        y_test_arr = y_test.to_numpy()

        feature_cols = FeatureEngineer.get_feature_columns(train_df)
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        results = []

        # 1. Baselines
        mean_model = HistoricalMeanDemandBaseline().fit(y_train)
        y_pred_mean = mean_model.predict(len(y_test))
        results.append({
            "model_name": "Historical Mean Baseline",
            "model_type": "baseline",
            **calculate_forecast_metrics(y_test_arr, y_pred_mean)
        })

        naive_model = NaiveDemandBaseline().fit(y_train)
        y_pred_naive = naive_model.predict(len(y_test))
        results.append({
            "model_name": "Naive Baseline (Last Value)",
            "model_type": "baseline",
            **calculate_forecast_metrics(y_test_arr, y_pred_naive)
        })

        s_naive_model = SeasonalNaiveBaseline().fit(y_train)
        y_pred_snaive = s_naive_model.predict(len(y_test))
        results.append({
            "model_name": "Seasonal Naive Baseline (7-Day)",
            "model_type": "baseline",
            **calculate_forecast_metrics(y_test_arr, y_pred_snaive)
        })

        # 2. ML Models
        ml_models = {
            "Linear Ridge Regression": LinearRidgeDemandModel(),
            "Log-Linear Econometric Model": LogLinearDemandModel(),
            "Random Forest Regressor": RandomForestDemandModel(n_estimators=40, max_depth=6),
            "Gradient Boosting Regressor": GradientBoostingDemandModel(max_iter=50, max_depth=5)
        }

        for name, model in ml_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results.append({
                    "model_name": name,
                    "model_type": "machine_learning",
                    **calculate_forecast_metrics(y_test_arr, y_pred)
                })
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        # Sort by lowest MAE
        results_df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
        best_model_name = results_df.iloc[0]["model_name"]

        return {
            "product_id": product_id,
            "train_obs": len(train_df),
            "test_obs": len(test_df),
            "cutoff_date": str(test_df["date"].min().date()),
            "best_model": best_model_name,
            "benchmark_table": results_df.to_dict(orient="records")
        }


class AblationExperiment:
    """Component-removal experiment evaluating the decision value of ML pricing vs simpler policies."""

    @classmethod
    def run_pricing_ablation(
        cls,
        demand_fn_ml,
        current_price: float,
        unit_cost: float,
        elasticity_val: float = -1.5,
        historical_mean_demand: float = 40.0
    ) -> List[Dict[str, Any]]:
        """Compares optimal pricing decisions derived under alternative demand formulations:
        1. Full ML Demand Model (non-linear, lagged features, price curve)
        2. Constant Elasticity Approximation (Iso-elastic assumption)
        3. Simple Cost-Plus Markup (+30% rule)
        4. Inelastic / Historical Mean Demand (Price-insensitive)
        """
        ablation_results = []

        base_constraints = PricingConstraints(
            current_price=current_price,
            unit_cost=unit_cost,
            min_price=round(current_price * 0.5, 2),
            max_price=round(current_price * 1.8, 2),
            max_price_change_pct=0.35,
            min_gross_margin=0.15
        )

        # 1. Full ML Model
        opt_ml = PriceOptimizer.optimize(demand_fn_ml, base_constraints, objective_type="profit")
        p_ml = opt_ml["recommended_price"]
        m_ml = PricingObjective.calculate_metrics(p_ml, float(demand_fn_ml(np.array([p_ml]))[0]), unit_cost)
        ablation_results.append({
            "component": "Full ML Demand Model",
            "description": "Non-linear ML regression with cross-features & historical lags",
            "recommended_price": p_ml,
            "price_delta_pct": opt_ml["price_delta_pct"],
            "expected_demand": round(m_ml["expected_demand"], 1),
            "expected_profit": round(m_ml["profit"], 2),
            "expected_margin": round(m_ml["margin"] * 100.0, 1),
            "decision_benefit": "Captures non-linear elasticity and dynamic temporal patterns"
        })

        # 2. Constant Elasticity (Ablation: Remove ML features, use constant elasticity)
        q0 = float(demand_fn_ml(np.array([current_price]))[0])
        def iso_elastic_fn(p_arr):
            return q0 * ((p_arr / current_price) ** elasticity_val)

        opt_iso = PriceOptimizer.optimize(iso_elastic_fn, base_constraints, objective_type="profit")
        p_iso = opt_iso["recommended_price"]
        # Ground-truth evaluation using the true demand environment
        m_iso_true = PricingObjective.calculate_metrics(p_iso, float(demand_fn_ml(np.array([p_iso]))[0]), unit_cost)
        profit_loss_iso = round(m_iso_true["profit"] - m_ml["profit"], 2)
        ablation_results.append({
            "component": "Ablation: Constant Elasticity Rule",
            "description": f"Fixed elasticity ({elasticity_val}) without dynamic features",
            "recommended_price": p_iso,
            "price_delta_pct": round((p_iso - current_price) / current_price * 100.0, 1),
            "expected_demand": round(m_iso_true["expected_demand"], 1),
            "expected_profit": round(m_iso_true["profit"], 2),
            "expected_margin": round(m_iso_true["margin"] * 100.0, 1),
            "decision_benefit": f"Delta vs ML: £{profit_loss_iso:+.2f} profit"
        })

        # 3. Simple Cost-Plus Markup (+35% rule)
        p_cost_plus = round(unit_cost * 1.35, 2)
        p_cost_plus = max(base_constraints.min_price, min(base_constraints.max_price, p_cost_plus))
        m_cp_true = PricingObjective.calculate_metrics(p_cost_plus, float(demand_fn_ml(np.array([p_cost_plus]))[0]), unit_cost)
        profit_loss_cp = round(m_cp_true["profit"] - m_ml["profit"], 2)
        ablation_results.append({
            "component": "Ablation: Cost-Plus Markup Policy",
            "description": "Rigid 35% markup over unit cost regardless of elasticity",
            "recommended_price": p_cost_plus,
            "price_delta_pct": round((p_cost_plus - current_price) / current_price * 100.0, 1),
            "expected_demand": round(m_cp_true["expected_demand"], 1),
            "expected_profit": round(m_cp_true["profit"], 2),
            "expected_margin": round(m_cp_true["margin"] * 100.0, 1),
            "decision_benefit": f"Delta vs ML: £{profit_loss_cp:+.2f} profit"
        })

        # 4. Inelastic Historical Mean (Ablation: Assume demand is price-insensitive)
        def inelastic_fn(p_arr):
            return np.full_like(p_arr, historical_mean_demand)

        opt_inel = PriceOptimizer.optimize(inelastic_fn, base_constraints, objective_type="profit")
        p_inel = opt_inel["recommended_price"]
        m_inel_true = PricingObjective.calculate_metrics(p_inel, float(demand_fn_ml(np.array([p_inel]))[0]), unit_cost)
        profit_loss_inel = round(m_inel_true["profit"] - m_ml["profit"], 2)
        ablation_results.append({
            "component": "Ablation: Inelastic Naive Demand",
            "description": "Ignores price sensitivity; pushes price to upper constraint",
            "recommended_price": p_inel,
            "price_delta_pct": round((p_inel - current_price) / current_price * 100.0, 1),
            "expected_demand": round(m_inel_true["expected_demand"], 1),
            "expected_profit": round(m_inel_true["profit"], 2),
            "expected_margin": round(m_inel_true["margin"] * 100.0, 1),
            "decision_benefit": f"Delta vs ML: £{profit_loss_inel:+.2f} profit"
        })

        return ablation_results
