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

    # Get the predicted answer
    predicted_ids = logits.argmax(-1).item()
    answer = processor.convert_ids_to_tokens(predicted_ids)
    return answer
