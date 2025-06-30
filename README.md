# image-captioning-api
A FastAPI-based image captioning API using the BLIP model from Hugging Face.

# 🖼️ Image Captioning API with FastAPI & BLIP

This project is a simple API built with FastAPI that uses the [Salesforce/BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) model to generate captions for images.

## 🚀 Features
- Upload an image or provide a URL
- Get a caption describing the image
- Runs locally with FastAPI
- Uses Hugging Face Transformers

## 🛠️ Requirements

```bash
pip install fastapi uvicorn transformers torch pillow requests
