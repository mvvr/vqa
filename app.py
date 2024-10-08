from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

# Load pre-trained BLIP model and processor
model_name = 'Salesforce/blip-vqa-base'
model = BlipForConditionalGeneration.from_pretrained(model_name)
processor = BlipProcessor.from_pretrained(model_name)

def preprocess_image(image: Image.Image):
    """Preprocess the image for BLIP."""
    image = image.convert("RGB")
    return processor(images=image, return_tensors="pt")

def preprocess_question(question: str):
    """Preprocess the question for BLIP."""
    return processor(text=question, return_tensors="pt")

def get_vqa_answer(image, question):
    """Get the answer from BLIP."""
    image_tensor = preprocess_image(image)
    question_ids = preprocess_question(question)
    
    # Prepare inputs
    inputs = {
        'pixel_values': image_tensor['pixel_values'],
        'input_ids': question_ids['input_ids']
    }
    
    with torch.no_grad():
        outputs = model(**inputs)
        answer_ids = torch.argmax(outputs.logits, dim=-1)
        answer = processor.decode(answer_ids[0], skip_special_tokens=True)
    
    return answer

def main():
    st.title('Visual Question Answering (VQA) with BLIP Model')
    
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
