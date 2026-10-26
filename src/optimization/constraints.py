"""Business and operational pricing constraints and feasibility verification."""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PricingConstraints:
    """Configurable business constraints for dynamic price optimization."""
    current_price: float
    unit_cost: float
    min_price: float
    max_price: float
    max_price_change_pct: Optional[float] = 0.25  # e.g. 0.25 = +/- 25% max price change
    min_gross_margin: Optional[float] = 0.15     # e.g. 0.15 = 15% min gross margin
    inventory_capacity: Optional[float] = None  # Supply cap on feasible sales
    competitor_price: Optional[float] = None
    max_competitor_markup_pct: Optional[float] = None  # e.g. price <= competitor * (1 + markup)
    min_competitor_discount_pct: Optional[float] = None # e.g. price >= competitor * (1 - discount)

    def validate_inputs(self) -> Tuple[bool, List[str]]:
        """Validates that constraint parameters are logically and mathematically consistent."""
        errors = []
        if self.min_price < 0:
            errors.append(f"Minimum price ({self.min_price}) cannot be negative.")
        if self.max_price < self.min_price:
            errors.append(f"Maximum price ({self.max_price}) cannot be less than minimum price ({self.min_price}).")
        if self.unit_cost < 0:
            errors.append(f"Unit cost ({self.unit_cost}) cannot be negative.")
        if self.current_price <= 0:
            errors.append(f"Current price ({self.current_price}) must be positive.")
        if self.max_price_change_pct is not None and self.max_price_change_pct <= 0:
            errors.append(f"Maximum price change percentage ({self.max_price_change_pct}) must be positive.")
        if self.min_gross_margin is not None and self.min_gross_margin >= 1.0:
            errors.append(f"Minimum gross margin ({self.min_gross_margin}) must be less than 100% (1.0).")
        return len(errors) == 0, errors

    def get_effective_price_bounds(self) -> Tuple[float, float, List[str]]:
        """Calculates the intersection of all boundary constraints and reports conflicts."""
        effective_min = self.min_price
        effective_max = self.max_price
        conflict_notes = []

        # Margin constraint: (P - c) / P >= margin => P >= c / (1 - margin)
        if self.min_gross_margin is not None and self.min_gross_margin < 1.0:
            margin_floor = self.unit_cost / (1.0 - self.min_gross_margin)
            effective_min = max(effective_min, margin_floor)

        # Max price change bounds: [P0 * (1 - chg), P0 * (1 + chg)]
        if self.max_price_change_pct is not None:
            change_floor = self.current_price * (1.0 - self.max_price_change_pct)
            change_ceiling = self.current_price * (1.0 + self.max_price_change_pct)
            effective_min = max(effective_min, change_floor)
            effective_max = min(effective_max, change_ceiling)

        # Competitor markup limit
        if self.competitor_price is not None and self.max_competitor_markup_pct is not None:
            comp_ceiling = self.competitor_price * (1.0 + self.max_competitor_markup_pct)
            effective_max = min(effective_max, comp_ceiling)

        # Competitor discount floor
        if self.competitor_price is not None and self.min_competitor_discount_pct is not None:
            comp_floor = self.competitor_price * (1.0 - self.min_competitor_discount_pct)
            effective_min = max(effective_min, comp_floor)

        if effective_min > effective_max:
            conflict_notes.append(
                f"Conflicting constraints: Effective lower bound ({effective_min:.2f}) exceeds upper bound ({effective_max:.2f})."
            )

        return float(effective_min), float(effective_max), conflict_notes

    def check_feasibility(self, price: float) -> Tuple[bool, List[str]]:
        """Checks if a candidate price satisfies every enabled constraint."""
        violations = []

        if price < self.min_price:
            violations.append(f"Price {price:.2f} < Min price {self.min_price:.2f}")
        if price > self.max_price:
            violations.append(f"Price {price:.2f} > Max price {self.max_price:.2f}")

        # Margin
        margin = (price - self.unit_cost) / price if price > 0 else 0.0
        if self.min_gross_margin is not None and margin < self.min_gross_margin - 1e-5:
            violations.append(f"Margin {margin:.1%} < Min required {self.min_gross_margin:.1%}")

        # Price change
        if self.max_price_change_pct is not None:
            change_pct = abs(price - self.current_price) / self.current_price
            if change_pct > self.max_price_change_pct + 1e-5:
                violations.append(f"Price change {change_pct:.1%} > Allowed limit {self.max_price_change_pct:.1%}")

        # Competitor
        if self.competitor_price is not None and self.max_competitor_markup_pct is not None:
            comp_ceiling = self.competitor_price * (1.0 + self.max_competitor_markup_pct)
            if price > comp_ceiling + 1e-5:
                violations.append(f"Price {price:.2f} > Competitor markup ceiling {comp_ceiling:.2f}")

        return len(violations) == 0, violations
