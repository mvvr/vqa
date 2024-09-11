import streamlit as st
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Load pre-trained LXMERT model and tokenizer
model_name = 'allenai/lxmert-base-uncased'
model = LxmertForQuestionAnswering.from_pretrained(model_name)
tokenizer = LxmertTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Resize image to a fixed size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def preprocess_image(image: Image.Image):
    """Convert image to tensor and extract visual features."""
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def preprocess_question(question: str):
    """Tokenize the question."""
    return tokenizer(question, return_tensors="pt")

def get_vqa_answer(image, question):
    """Get the answer from LXMERT."""
    image_tensor = preprocess_image(image)
    question_ids = preprocess_question(question)
    
    # Prepare inputs
    inputs = {
        'input_ids': question_ids['input_ids'],
        'attention_mask': question_ids['attention_mask'],
        'visual_feats': image_tensor,
        'visual_pos': torch.zeros((1, image_tensor.size(2), image_tensor.size(3))).long()  # Placeholder for positional embeddings
    }
    
    with torch.no_grad():
        outputs = model(**inputs)
        answer_ids = torch.argmax(outputs.logits, dim=-1)
        answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    
    return answer

def main():
    st.title('Visual Question Answering with LXMERT')
    
    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Input question
    question = st.text_input("Enter your question")
    
    if st.button("Get Answer"):
        if uploaded_image and question:
            try:
                answer = get_vqa_answer(image, question)
                st.write(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload an image and enter a question.")

if __name__ == "__main__":
    main()
