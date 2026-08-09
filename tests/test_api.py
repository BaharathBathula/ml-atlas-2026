from fastapi.testclient import TestClient

from api.main import app, service


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "loaded" in response.json()


def test_predict_endpoint_when_demo_model_present():
    client = TestClient(app)

    if not service.is_loaded:
        response = client.post("/predict", json={
            "records": [{
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            }]
        })
        assert response.status_code == 503
        return

    response = client.post("/predict", json={
        "records": [{
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }]
    })
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 1
