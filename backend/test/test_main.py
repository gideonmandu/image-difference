# from fastapi.testclient import TestClient
import pytest
from httpx import AsyncClient
from src.main import app


@pytest.mark.anyio
async def test_index():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/file")
    assert response.status_code == 200
    assert response.json() == {"test": "hello world"}


@pytest.mark.anyio
async def test_create_upload_file_no_file():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/file/upload")
    assert response.status_code == 200
    assert response.json() == {"message": "No upload file sent"}


@pytest.mark.anyio
async def test_create_upload_file():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/file/upload", files="./files/test10.jpg")
    assert response.status_code == 200
    assert response.json() == {
        "imagetype": "passport",
    }
