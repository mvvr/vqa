from transformers import ViltProcessor
from PIL import Image

# Initialize processor globally
processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')

def preprocess_image(image: Image.Image):
    # Convert image to required format and size
    return processor(images=image, return_tensors="pt").pixel_values

def preprocess_question(question: str):
    # Tokenize and preprocess the question
    return processor(text=question, return_tensors="pt").input_ids
