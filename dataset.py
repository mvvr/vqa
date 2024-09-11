import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VQADataset(Dataset):
    def __init__(self, image_paths, questions, answers, tokenizer):
        self.image_paths = image_paths
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        
        question = self.tokenizer(self.questions[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=32)
        question_ids = question['input_ids'].squeeze(0)
        attention_mask = question['attention_mask'].squeeze(0)
        
        # Assume answers are encoded as integers
        answer = torch.tensor(self.answers[idx], dtype=torch.long)
        
        return image, question_ids, attention_mask, answer
