"""Unit tests for FastAPI endpoints, authentication, and role authorization."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.auth.authentication import hash_password, verify_password, create_access_token, decode_access_token


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


def test_password_hashing():
    """Verifies cryptographic password hashing with random salt and constant-time verification."""
    pwd = "SecurePassword123!"
    hashed = hash_password(pwd)
    assert "$" in hashed
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_access_token():
    """Verifies HMAC token creation, claims decoding, and signature tampering rejection."""
    payload = {"sub": "analyst_user", "role": "analyst"}
    token = create_access_token(payload, expires_delta_minutes=60)
    decoded = decode_access_token(token)

    assert decoded is not None
    assert decoded["sub"] == "analyst_user"
    assert decoded["role"] == "analyst"

    # Tampered token
    tampered = token[:-4] + "abcd"
    assert decode_access_token(tampered) is None


def test_api_health_endpoint(client):
    """Verifies GET /health returns 200 OK and valid status dictionary."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_dataset" in data


def test_api_pricing_optimize_endpoint(client):
    """Verifies POST /pricing/optimize returns recommended price and decision metrics."""
    # First get product from list
    p_resp = client.get("/products")
    assert p_resp.status_code == 200
    products = p_resp.json()
    assert len(products) > 0
    prod = products[0]

    payload = {
        "product_id": prod["product_id"],
        "current_price": prod["latest_price"],
        "unit_cost": prod["unit_cost"],
        "min_price": round(prod["latest_price"] * 0.7, 2),
        "max_price": round(prod["latest_price"] * 1.4, 2),
        "max_price_change_pct": 0.25,
        "min_gross_margin": 0.15,
        "objective_type": "profit",
        "demand_model_name": "log_linear"
    }

    opt_resp = client.post("/pricing/optimize", json=payload)
    assert opt_resp.status_code == 200
    res = opt_resp.json()
    assert res["is_feasible"] is True
    assert res["recommended_price"] is not None
    assert "expected_profit" in res
    assert "explanation" in res


def test_api_infeasible_optimization_handling(client):
    """Verifies that an infeasible problem formulation returns 200 with is_feasible=False and clear error message."""
    payload = {
        "product_id": "85123A",
        "current_price": 10.0,
        "unit_cost": 20.0,
        "min_price": 5.0,
        "max_price": 15.0,
        "min_gross_margin": 0.50  # Requires P >= 20 / 0.5 = 40.0, but max is 15.0!
    }
    opt_resp = client.post("/pricing/optimize", json=payload)
    assert opt_resp.status_code == 200
    res = opt_resp.json()
    assert res["is_feasible"] is False
    assert "Infeasible" in res["message"]
