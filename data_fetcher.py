import torchvision.transforms as transforms
from transformers import BertTokenizer
from PIL import Image

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def preprocess_question(question: str, tokenizer, max_length=20):
    return tokenizer(question, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
