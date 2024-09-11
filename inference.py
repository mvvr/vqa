from transformers import ViltForQuestionAnswering, ViltProcessor
#from PIL import Image
import torch
from data_fetcher import preprocess_image, preprocess_question
# Load pre-trained model and tokenizer
# Load a pre-trained VQA model and tokenizer
model_name = 'dandelin/vilt-b32-finetuned-vqa'  # Use a specific pre-trained VQA model
model = ViltForQuestionAnswering.from_pretrained(model_name)
tokenizer = ViltProcessor.from_pretrained(model_name)


#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def get_vqa_answer(image, question):
    image_tensor = preprocess_image(image)
    question_tensor = preprocess_question(question, tokenizer)

    with torch.no_grad():
        output = model(image_tensor, question_tensor)

    # Get the predicted answer
    _, predicted_idx = torch.max(output, dim=1)
    answer = answer_vocab[predicted_idx.item()]  # Use a predefined answer vocabulary
    return answer
