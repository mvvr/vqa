import streamlit as st
from PIL import Image
from inference import get_vqa_answer

# Streamlit UI
st.title("Visual Question Answering (VQA) System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Ask a question about the image")

if uploaded_file and question:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Get answer from VQA system
    answer = get_vqa_answer(image, question)
    st.write(f"Answer: {answer}")
