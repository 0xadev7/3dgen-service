from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_generate_endpoint():
    r = client.post("/generate/", data={"prompt": "a test cube"})
    assert r.status_code == 200
    assert isinstance(r.content, (bytes, bytearray))

def test_preview_endpoint():
    r = client.post("/preview.png", data={"prompt": "a test cube"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
