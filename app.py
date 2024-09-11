import torch
from PIL import Image
from transformers import OscarForQuestionAnswering, OscarTokenizer
import streamlit as st

# Load pre-trained OSCAR model and tokenizer
model_name = 'microsoft/oscar-base-vqav2'
model = OscarForQuestionAnswering.from_pretrained(model_name)
tokenizer = OscarTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def preprocess_image(image: Image.Image):
    """Preprocess the image for OSCAR."""
    # For OSCAR, you typically need a feature extractor or pre-trained visual encoder
    # Assuming we have some `extract_features` function or you can use pre-trained feature extractors
    # For now, we'll return a dummy tensor for demonstration
    image = image.convert("RGB")
    # Placeholder: Implement actual feature extraction
    return torch.rand(1, 3, 224, 224)  # Example shape, adjust as needed

def preprocess_question(question: str):
    """Tokenize the question for OSCAR."""
    return tokenizer(question, return_tensors="pt")

def get_vqa_answer(image, question):
    """Get the answer from OSCAR."""
    image_tensor = preprocess_image(image)
    question_ids = preprocess_question(question)
    
    # Prepare inputs
    inputs = {
        'input_ids': question_ids['input_ids'],
        'visual_feats': image_tensor
    }
    
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], visual_feats=inputs['visual_feats'])
        answer_ids = torch.argmax(outputs.logits, dim=-1)
        answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    
    return answer

def main():
    st.title('Visual Question Answering (VQA) with OSCAR Model')
    
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
