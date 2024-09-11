import torch
from PIL import Image
from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
import streamlit as st

# Load pre-trained VQA model and tokenizer
model_name = 'uclanlp/visualbert-vqa-coco'
model = VisualBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = VisualBertTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def preprocess_image(image: Image.Image):
    """Convert image to tensor."""
    image = image.convert("RGB")
    # Placeholder function - actual image preprocessing required
    return image

def preprocess_question(question: str):
    """Tokenize the question."""
    return tokenizer(question, return_tensors="pt")

def get_vqa_answer(image, question):
    """Get the answer from VisualBERT."""
    image_tensor = preprocess_image(image)
    question_ids = preprocess_question(question)
    
    # Prepare inputs
    inputs = {
        'input_ids': question_ids['input_ids'],
        'visual_feats': image_tensor
    }
    
    with torch.no_grad():
        outputs = model(**inputs)
        answer = tokenizer.decode(outputs.logits.argmax(-1))
    
    return answer

def main():
    st.title('Visual Question Answering (VQA) with VisualBERT Model')
    
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
