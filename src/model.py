import torch
import torch.nn as nn
import torch.nn.functional as F

class HandwritingModel(nn.Module):
    def __init__(self, num_classes, input_dim=7, hidden_dim=128, struct_dim=128):
        """
        Structural-Aware Handwriting Similarity Model
        :param num_classes: Number of characters (classes)
        :param input_dim: Dimension of input features (default 7)
        :param hidden_dim: Dimension of hidden layers and embedding
        :param struct_dim: Dimension of structural embedding
        """
        super(HandwritingModel, self).__init__()
        
        # 1. 1D Convolutional Backbone
        # Input: (B, input_dim, L)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 2. GRU Layer
        # GRU input: (B, L, hidden_dim) -> (batch_first)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Heads
        # Embedding E is the output of Global Average Pooling
        
        # Character Classification Head
        self.fc_char = nn.Linear(hidden_dim, num_classes)
        
        # Structural Similarity Head
        # Projects to structural embedding space [0, 1]^R
        self.fc_struct = nn.Linear(hidden_dim, struct_dim)

    def forward(self, x):
        """
        :param x: Input tensor of shape (B, L, input_dim) -> adjusted to (B, input_dim, L) for Conv
        :return: char_logits, struct_embedding
        """
        # Permute for Conv1d: (B, L, D) -> (B, D, L)
        x = x.permute(0, 2, 1)
        
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Prepare for GRU: (B, D, L) -> (B, L, D)
        x = x.permute(0, 2, 1)
        
        # GRU
        # output: (B, L, hidden_dim), hn: (1, B, hidden_dim)
        # We can use the last hidden state or pool the outputs. 
        # Doc mentions Global Average Pooling, usually after RNN or Conv.
        # Let's pool the GRU outputs.
        out, _ = self.gru(x)
        
        # Global Average Pooling over time dimension
        # out: (B, L, hidden_dim) -> (B, hidden_dim)
        embedding = torch.mean(out, dim=1)
        
        # Heads
        char_logits = self.fc_char(embedding)
        
        # Sigmoid for structural embedding to range [0, 1]
        struct_pred = torch.sigmoid(self.fc_struct(embedding))
        
        return char_logits, struct_pred
