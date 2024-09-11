import torch
from transformers import ViltForQuestionAnswering, ViltProcessor
from data_fetcher import preprocess_image, preprocess_question

# Load pre-trained VQA model and processor
model_name = 'dandelin/vilt-b32-finetuned-vqa'
model = ViltForQuestionAnswering.from_pretrained(model_name)
processor = ViltProcessor.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def get_vqa_answer(image, question):
    # Preprocess the image and question
    image_tensor = preprocess_image(image)
    question_tensor = preprocess_question(question, processor)

    # Process inputs
    inputs = processor(images=image_tensor, text=question, return_tensors="pt", padding=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted answer
    answer = processor.convert_ids_to_tokens(logits.argmax(-1).item())
    return answer
