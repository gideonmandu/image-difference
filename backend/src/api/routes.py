import uuid
from typing import Optional
from fastapi import UploadFile, APIRouter
from src.services.image_processing import ImageProcessor

router = APIRouter(prefix="/file", tags=["passport & ID upload"])


# @router.get("",)
# async def index():
#     return {"test": "hello world"}


@router.post("/upload/")
async def create_upload_file(file: Optional[UploadFile] = None):
    if not file:
        return {"message": "No upload file sent"}
    file.filename = f"src/files/{file.filename}.{file.filename.split('.')[1]}"
    image_bytes = await file.read()

    with open(f"{file.filename}", "wb") as f:
        f.write(image_bytes)
    image_processor = ImageProcessor(image=f"{file.filename}")
    return {
        "filename": file.filename,
        "imagetype": image_processor.image_type(),
    }
