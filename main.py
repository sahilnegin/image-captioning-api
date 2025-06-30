from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests
from io import BytesIO
import os


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_image():
    choice = input("Choose image source: [1] Local file  [2] URL\nEnter 1 or 2: ")

    if choice == '1':
        image_path = input("Enter the full path to your local image file: ").strip('"')
        if not os.path.exists(image_path):
            print("❌ File not found.")
            exit()
        return Image.open(image_path).convert("RGB")

    elif choice == '2':
        image_url = input("Enter the image URL: ").strip()
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print("❌ Failed to load image from URL:", e)
            exit()
    else:
        print("❌ Invalid choice.")
        exit()


image = load_image()


inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("\n Generated Caption:", caption)
