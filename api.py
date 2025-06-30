from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
import requests
import torch

app = FastAPI()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_image_from_bytes(file_bytes):
    try:
        return Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
def load_image_from_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {e}")
    
@app.post("/caption")
async def generate_caption(
    image: UploadFile = File(None),
    image_url: str = Form(None)
):
    if image:
        image_data = await image.read()
        img = load_image_from_bytes(image_data)
    elif image_url:
        img = load_image_from_url(image_url)
    else:
        raise HTTPException(status_code=400, detail="No image file or URL provided.")

    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return {"caption": caption}