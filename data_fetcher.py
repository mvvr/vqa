from transformers import ViltProcessor
from PIL import Image

# Initialize processor globally
processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')

def preprocess_image(image: Image.Image):
    # Convert image to RGB and resize if necessary
    image = image.convert("RGB")
    # Use the processor to preprocess the image
    return processor(images=image, return_tensors="pt").pixel_values

def preprocess_question(question: str):
    # Use the processor to preprocess the question
    return processor(text=question, return_tensors="pt").input_ids
