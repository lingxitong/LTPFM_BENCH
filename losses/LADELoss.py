import torch
import torch.nn as nn
import numpy as np
#LADELoss:Label distribution decoupling loss.
class LADELoss(nn.Module):
    def __init__(self, cls_num_list, remine_lambda=0.1):
        super(LADELoss, self).__init__()

        self.num_classes = len(cls_num_list)
        self.remine_lambda = remine_lambda
        
        cls_num_tensor = torch.tensor(cls_num_list).float()
        prior = cls_num_tensor / cls_num_tensor.sum()
        balanced_prior = torch.tensor(1. / self.num_classes).float()
        cls_weight = cls_num_tensor / cls_num_tensor.sum()

        self.register_buffer('prior', prior)
        self.register_buffer('balanced_prior', balanced_prior)
        self.register_buffer('cls_weight', cls_weight)

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target):
        """
        y_pred: (Batch, Num_Classes) -> Logits
        target: (Batch) -> Labels
        """
        target = target.to(y_pred.device)
        one_hot = (target.unsqueeze(0) == torch.arange(self.num_classes).to(target.device).unsqueeze(1)) # (C, N)
        per_cls_pred_spread = y_pred.T * one_hot 

        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T 
        num_samples_per_cls = torch.sum(one_hot, dim=1).float()
        estim_loss, _, _ = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        loss = -torch.sum(estim_loss * self.cls_weight)
        
        return loss