from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import routes

api = FastAPI()


def configure_api(app: FastAPI) -> FastAPI:
    origins = [
        "http://localhost",
        "http://localhost:5000",
        "http://localhost:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes.router)
    return app


app = configure_api(app=api)
