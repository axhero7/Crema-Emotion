import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification
import optuna

class Whisper_NET(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, hidden_sizes):
        super(Whisper_NET, self).__init__()
        self.base_model = AutoModelForAudioClassification.from_pretrained(pretrained_model_name)
        
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        
        base_output_dim = self.base_model.classifier.out_features
        
        layers = []
        input_dim = base_output_dim
        for hidden_size in range(hidden_sizes):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.custom_layers = nn.Sequential(*layers)

    def forward(self, input_values, attention_mask=None):
        base_output = self.base_model(input_values, attention_mask=attention_mask)
        return self.custom_layers(base_output.logits)