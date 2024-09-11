import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_fetcher import preprocess_image, preprocess_question
from models import VQAModel
from transformers import BertTokenizer

# Define dataset
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, questions, answers, tokenizer):
        self.image_paths = image_paths
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = preprocess_image(self.image_paths[idx])
        question = preprocess_question(self.questions[idx], self.tokenizer)
        answer = torch.tensor(self.answers[idx])
        return image, question, answer

# Training function
def train_model(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, questions, answers in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, questions)
            loss = criterion(outputs, answers)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'vqa_model.pth')

if __name__ == "__main__":
    # Hyperparameters
    num_answers = 1000  # Define based on dataset
    batch_size = 32
    learning_rate = 0.001

    # Initialize model, tokenizer, and optimizer
    model = VQAModel(num_answers)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Load dataset and dataloader
    dataset = VQADataset(image_paths, questions, answers, tokenizer)  # image_paths, questions, and answers should be lists
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_model(model, dataloader, optimizer, criterion, epochs=10)
