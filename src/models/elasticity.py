"""Econometric Price Elasticity of Demand estimation using log-log regression."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats
from src.utils.logger import logger


class PriceElasticityEstimator:
    """Estimates empirical price elasticity using an econometric log-log model: ln(Q) = beta_0 + beta_1 * ln(P) + gamma * X."""

    MIN_RECORDS_REQUIRED = 10
    MIN_PRICE_CV = 0.04  # Minimum coefficient of variation for price

    @classmethod
    def estimate_elasticity(
        cls, 
        df: pd.DataFrame, 
        product_id: str,
        control_cols: Optional[List[str]] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Estimates empirical price elasticity with standard errors, confidence intervals, and stability diagnostics."""
        # Filter for the target product
        prod_data = df[df["product_id"] == str(product_id)].copy()
        
        # Filter for strictly positive demand and price
        prod_data = prod_data[(prod_data["quantity"] > 0) & (prod_data["price"] > 0)].dropna(subset=["price", "quantity"])
        n_obs = len(prod_data)

        prices = prod_data["price"].to_numpy()
        quantities = prod_data["quantity"].to_numpy()

        # Check sample size
        if n_obs < cls.MIN_RECORDS_REQUIRED:
            return {
                "product_id": product_id,
                "is_reliable": False,
                "sample_size": n_obs,
                "warning": f"Insufficient sample size ({n_obs} observations < {cls.MIN_RECORDS_REQUIRED} required).",
                "elasticity": None,
                "is_assumed": False,
                "price_cv": None
            }

        # Check price variation: CV = std / mean
        price_mean = float(np.mean(prices))
        price_std = float(np.std(prices))
        price_cv = price_std / price_mean if price_mean > 0 else 0.0

        if price_cv < cls.MIN_PRICE_CV:
            return {
                "product_id": product_id,
                "is_reliable": False,
                "sample_size": n_obs,
                "price_cv": round(price_cv, 4),
                "price_range": [round(float(np.min(prices)), 2), round(float(np.max(prices)), 2)],
                "warning": (
                    f"Insufficient historical price variation (CV = {price_cv:.2%} < {cls.MIN_PRICE_CV:.2%}). "
                    "Empirical elasticity cannot be reliably identified. Recommend using an assumed elasticity value."
                ),
                "elasticity": None,
                "is_assumed": False
            }

        # Log transformations
        log_q = np.log(quantities)
        log_p = np.log(prices)

        # Build design matrix with intercept and optional controls
        X_matrix = [np.ones(n_obs), log_p]
        col_names = ["const", "log_price"]

        if control_cols:
            for col in control_cols:
                if col in prod_data.columns and prod_data[col].nunique() > 1:
                    X_matrix.append(prod_data[col].to_numpy())
                    col_names.append(col)

        X = np.column_stack(X_matrix)
        y = log_q

        # Econometric OLS estimation
        try:
            # Beta = (X^T X)^-1 X^T y
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta = XtX_inv @ X.T @ y
            residuals = y - X @ beta
            dof = n_obs - X.shape[1]

            if dof <= 0:
                raise ValueError("Degrees of freedom <= 0")

            sigma2 = float(np.sum(residuals ** 2) / dof)
            vcov = sigma2 * XtX_inv
            se_beta = np.sqrt(np.diag(vcov))

            # Beta 1 is the price elasticity coefficient
            elasticity_est = float(beta[1])
            elasticity_se = float(se_beta[1])

            # t-statistic and p-value
            t_stat = elasticity_est / elasticity_se if elasticity_se > 0 else 0.0
            p_val = float(2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof)))

            # Confidence interval
            crit_val = stats.t.ppf((1 + confidence_level) / 2, df=dof)
            ci_lower = elasticity_est - crit_val * elasticity_se
            ci_upper = elasticity_est + crit_val * elasticity_se

            # R-squared
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # Economic plausibility check
            is_plausible = elasticity_est < 0.0
            warning_msg = None
            if not is_plausible:
                warning_msg = (
                    f"Positive elasticity ({elasticity_est:.3f}) detected! Standard economic demand is downward-sloping. "
                    "This positive relationship likely stems from simultaneous promotional demand spikes or luxury goods."
                )

            # Sub-sample stability check
            split_idx = n_obs // 2
            X1, y1 = X[:split_idx], y[:split_idx]
            X2, y2 = X[split_idx:], y[split_idx:]
            b1 = (np.linalg.pinv(X1.T @ X1) @ X1.T @ y1)[1] if len(y1) > X.shape[1] else elasticity_est
            b2 = (np.linalg.pinv(X2.T @ X2) @ X2.T @ y2)[1] if len(y2) > X.shape[1] else elasticity_est
            stability_gap = float(abs(b1 - b2))

            return {
                "product_id": product_id,
                "is_reliable": True,
                "is_assumed": False,
                "sample_size": n_obs,
                "elasticity": round(elasticity_est, 4),
                "std_error": round(elasticity_se, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "confidence_level": confidence_level,
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 4),
                "r_squared": round(r_squared, 4),
                "price_cv": round(price_cv, 4),
                "price_range": [round(float(np.min(prices)), 2), round(float(np.max(prices)), 2)],
                "controls_used": col_names[2:],
                "stability_check": {
                    "split_1_elasticity": round(float(b1), 4),
                    "split_2_elasticity": round(float(b2), 4),
                    "stability_gap": round(stability_gap, 4)
                },
                "is_economically_plausible": is_plausible,
                "warning": warning_msg,
                "observational_disclaimer": (
                    "Observational Data Limitation: This elasticity estimate represents statistical association. "
                    "In observational retail data, price changes often coincide with demand shocks or marketing campaigns, "
                    "potentially causing omitted variable bias and endogeneity. Not a causal randomized trial."
                )
            }
        except Exception as e:
            logger.error(f"Elasticity estimation failed for {product_id}: {e}")
            return {
                "product_id": product_id,
                "is_reliable": False,
                "is_assumed": False,
                "sample_size": n_obs,
                "warning": f"Numerical error during regression: {str(e)}",
                "elasticity": None
            }

    @classmethod
    def get_assumed_elasticity_fallback(
        cls, 
        product_id: str, 
        assumed_value: float = -1.5, 
        rationale: str = "Industry standard assumption for discretionary retail"
    ) -> Dict[str, Any]:
        """Provides an explicitly labeled assumption-based elasticity object."""
        return {
            "product_id": product_id,
            "is_reliable": True,
            "is_assumed": True,
            "sample_size": 0,
            "elasticity": round(assumed_value, 4),
            "std_error": 0.25,
            "ci_lower": round(assumed_value - 0.49, 4),
            "ci_upper": round(assumed_value + 0.49, 4),
            "confidence_level": 0.95,
            "rationale": rationale,
            "warning": "ASSUMPTION-BASED: Not derived from historical econometric regression."
        }
