import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50

class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super(VQAModel, self).__init__()
        
        # Image feature extractor (ResNet50 pre-trained)
        self.cnn = resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Text feature extractor (BERT pre-trained)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Fusion Layer (combining image and text features)
        self.fc_fusion = nn.Linear(2048 + 768, 1024)
        self.fc_out = nn.Linear(1024, num_answers)
        
        # Non-linearities
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, image_features, question_tokens):
        # Extract image features using CNN
        img_feats = self.cnn(image_features)
        
        # Extract question features using BERT
        question_outputs = self.bert(**question_tokens)
        question_feats = question_outputs.pooler_output
        
        # Concatenate image and text features
        combined_feats = torch.cat([img_feats, question_feats], dim=1)
        
        # Fusion and output layer
        x = self.relu(self.fc_fusion(combined_feats))
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return x
