import torch
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor
import streamlit as st

# Load pre-trained VQA model and processor
model_name = 'dandelin/vilt-b32-finetuned-vqa'
model = ViltForQuestionAnswering.from_pretrained(model_name)
processor = ViltProcessor.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def preprocess_image(image: Image.Image):
    """Preprocess the image for the model."""
    # Convert image to RGB
    image = image.convert("RGB")
    # Use ViltProcessor to handle the image
    return processor(images=image, return_tensors="pt")['pixel_values']

def preprocess_question(question: str):
    """Preprocess the question for the model."""
    return processor(text=question, return_tensors="pt")['input_ids']

def get_vqa_answer(image, question):
    """Get the answer from the model."""
    # Preprocess the image and question
    image_tensor = preprocess_image(image)
    question_ids = preprocess_question(question)

    # Prepare inputs for the model
    inputs = {
        'pixel_values': image_tensor,
        'input_ids': question_ids
    }

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Decode the answer
    answer_ids = logits.argmax(-1).tolist()
    answer = processor.convert_ids_to_tokens(answer_ids[0])
    return answer

def main():
    st.title('Visual Question Answering (VQA) with VILT Model')
    
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
