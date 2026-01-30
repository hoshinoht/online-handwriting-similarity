import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, lambda_struct=0.2):
        """
        Combined Loss: CrossEntropy + Lambda * (1 - CosineSimilarity)
        :param lambda_struct: Weight for structural loss
        """
        super(CombinedLoss, self).__init__()
        self.lambda_struct = lambda_struct
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, char_logits, char_targets, struct_pred, struct_target):
        """
        :param char_logits: (B, C)
        :param char_targets: (B,)
        :param struct_pred: (B, R)
        :param struct_target: (B, R)
        :return: total_loss, loss_dict
        """
        # Character Class Loss
        loss_char = self.ce_loss(char_logits, char_targets)
        
        # Structural Similarity Loss
        # Cosine Similarity: sum(a*b)/(|a|*|b|)
        # F.cosine_similarity returns similarity for each element in batch
        sim_struct = F.cosine_similarity(struct_pred, struct_target, dim=1)
        
        # Loss = 1 - Similarity (mean over batch)
        loss_struct = 1.0 - torch.mean(sim_struct)
        
        # Total Loss
        total_loss = loss_char + self.lambda_struct * loss_struct
        
        return total_loss, {
            "loss_char": loss_char.item(),
            "loss_struct": loss_struct.item(),
            "total_loss": total_loss.item()
        }
