import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: (B, L, hidden_dim)
        # weights: (B, L, 1)
        weights = F.softmax(self.attention(x), dim=1)
        # pooled: (B, hidden_dim)
        pooled = torch.sum(x * weights, dim=1)
        return pooled, weights

class HandwritingModel(nn.Module):
    def __init__(self, num_classes, input_dim=7, hidden_dim=256, struct_dim=128, dropout=0.3):
        """
        Deep Structural-Aware Handwriting Similarity Model (ResNet1D + BiGRU + Attention)
        """
        super(HandwritingModel, self).__init__()
        
        # 1. ResNet-1D Backbone
        # Input: (B, input_dim, L)
        self.start_conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn_start = nn.BatchNorm1d(64)
        
        self.layer1 = ResidualBlock(64, 128, dropout=dropout)
        self.layer2 = ResidualBlock(128, 256, dropout=dropout)
        self.layer3 = ResidualBlock(256, hidden_dim, dropout=dropout)
        
        # 2. Bidirectional GRU
        # Input to GRU: (B, L, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
        # BiGRU output dim = hidden_dim * 2
        
        # 3. Attention
        self.attention = Attention(hidden_dim * 2)
        
        # 4. Heads
        # Context vector from attention is (B, hidden_dim*2)
        
        # Character Classification Head
        self.fc_char = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Structural Similarity Head
        self.fc_struct = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, struct_dim)
        )

    def forward(self, x):
        """
        :param x: (B, L, input_dim) -> (B, input_dim, L) for Conv
        """
        x = x.permute(0, 2, 1)
        
        # Backbone
        x = F.relu(self.bn_start(self.start_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Prepare for GRU: (B, D, L) -> (B, L, D)
        x = x.permute(0, 2, 1)
        
        # GRU
        # out: (B, L, hidden_dim * 2)
        out, _ = self.gru(x)
        
        # Attention Pooling
        embedding, attn_weights = self.attention(out)
        
        # Heads
        char_logits = self.fc_char(embedding)
        struct_pred = torch.sigmoid(self.fc_struct(embedding))
        
        return char_logits, struct_pred
