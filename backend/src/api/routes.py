from typing import Optional
from fastapi import UploadFile, File, APIRouter
from src.services.image_processing import ImageProcessor

router = APIRouter(prefix='/file',tags=['passport & ID upload'])


@router.get('')
async def test():
    return {"test":"nada"}


@router.post("/upload/")
async def create_upload_file(file: Optional[UploadFile] = File(None)):
    if not file:
        return {"message": "No upload file sent"}
    print(file)
    image = file.file.read()
    # TODO fix image being read
    image_processor = ImageProcessor(image=image)
    return {"filename": file.filename, "Imagetype":image_processor.image_type()}