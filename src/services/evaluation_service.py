"""Evaluation service for model benchmarking and component ablation experiments."""
from typing import Dict, Any, List
from src.services.demand_service import demand_service
from src.evaluation.experiments import ModelBenchmarkExperiment, AblationExperiment


class EvaluationService:
    """Service providing chronological model benchmarking and ablation studies."""

    @classmethod
    def get_model_benchmark(cls, product_id: str) -> Dict[str, Any]:
        """Runs chronological validation on the product demand time series."""
        _, feat_df = demand_service.get_processed_data()
        return ModelBenchmarkExperiment.run_benchmark(feat_df, product_id=product_id)

    @classmethod
    def get_ablation_study(
        cls, 
        product_id: str, 
        current_price: float, 
        unit_cost: float,
        demand_model_name: str = "log_linear"
    ) -> List[Dict[str, Any]]:
        """Runs component ablation study comparing ML pricing vs simplified heuristics."""
        demand_fn = demand_service.get_demand_curve_evaluator(product_id, demand_model_name)
        daily_df, _ = demand_service.get_processed_data()
        prod_data = daily_df[daily_df["product_id"] == str(product_id)]
        mean_demand = float(prod_data["quantity"].mean()) if len(prod_data) > 0 else 30.0

        return AblationExperiment.run_pricing_ablation(
            demand_fn_ml=demand_fn,
            current_price=current_price,
            unit_cost=unit_cost,
            historical_mean_demand=mean_demand
        )
