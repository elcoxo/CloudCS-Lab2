# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"IE": 1.0,
                "NS": 1.0,
                "TF": 0.0,
                "JP": 0.0
                }

    def mock_load_model(*args, **kwargs) -> None:
        return None

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)

    from main import app
    return TestClient(app)


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_correctness(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json={"text": 'hi my name is slim shady'}
    )
    json_response = {"IE": 1.0,
                     "NS": 1.0,
                     "TF": 0.0,
                     "JP": 0.0
                     }
    assert response.status_code == 200


def test_token_not_correctness(init_test_client):
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer kedjkj"},
        json={"text": 'hi my name is slim shady'}
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_token_absent(init_test_client):
    response = init_test_client.post(
        "/predictions",
        json={"text": 'hi my name is slim shady'}
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated"
    }


def test_inference(init_test_client):
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json={"text": 'hi my name is slim shady'}
    )
    assert response.status_code == 200
    assert response.json()["IE"] == 1.0
    assert response.json()["NS"] == 1.0
    assert response.json()["TF"] == 0.0
    assert response.json()["JP"] == 0.0
