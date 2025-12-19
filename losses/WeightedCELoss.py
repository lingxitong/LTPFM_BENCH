import torch
import torch.nn as nn
import numpy as np

class WeightedCELoss(nn.Module):
    def __init__(self, cls_num_list, mode='cb_weight'):
        super(WeightedCELoss, self).__init__()

        if isinstance(cls_num_list, list):
            cls_num_list = np.array(cls_num_list)
            
        if mode == 'cb_weight':
            print("🚀 Using Class-Balanced Weighting (RWLoss_1)")
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            weights = torch.FloatTensor(per_cls_weights)
            
        elif mode == 'log_weight':
            print("🚀 Using Log-Frequency Weighting (RWLoss_2)")
            num = sum(cls_num_list)
            prob = cls_num_list / num
            prob = torch.FloatTensor(prob)
            max_prob = prob.max()
            prob = prob / max_prob
            weights = -prob.log() + 1
            
        else:
            print("🚀 Using Simple Inverse Frequency Weighting")
            weights = 1.0 / (torch.tensor(cls_num_list).float() + 1e-6)
            weights = weights / weights.sum() * len(cls_num_list)

        self.register_buffer('weights', weights)

    def forward(self, logits, targets):
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
            
        return nn.functional.cross_entropy(logits, targets, weight=self.weights)