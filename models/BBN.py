import torch
import torch.nn as nn
import copy

class BBN_Adapter(nn.Module):
    def __init__(self, in_dim, num_classes, MIL_Name='AB_MIL', topk=64):
        super(BBN_Adapter, self).__init__()
        
        self.agg_feature_dim = 0
        
        if MIL_Name == 'AB_MIL':
            from modules.AB_MIL.ab_mil import AB_MIL
            self.MIL = AB_MIL(in_dim=in_dim, num_classes=num_classes, dropout=0.1)
            self.agg_feature_dim = 512 # AB_MIL 的 L
        elif MIL_Name == 'AMD_MIL':
            from modules.AMD_MIL.amd_mil import AMD_MIL
            self.MIL = AMD_MIL(in_dim=in_dim,num_classes=num_classes,dropout=0.1,act=nn.ReLU(),embed_dim=512)
            self.agg_feature_dim = 512    
        elif MIL_Name == 'CLAM_SB_MIL':
            from modules.CLAM_SB_MIL.clam_sb_mil import CLAM_SB_MIL
            self.MIL = CLAM_SB_MIL(in_dim=in_dim, num_classes=num_classes, dropout=0.1)
            self.agg_feature_dim = 512
        elif MIL_Name == 'CLAM_MB_MIL':
            from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL
            self.MIL = CLAM_MB_MIL(embed_dim=in_dim, num_classes=num_classes, dropout=0.1)
            self.agg_feature_dim = 512
        elif MIL_Name == 'TRANS_MIL':
            from modules.TRANS_MIL.trans_mil import TRANS_MIL
            self.MIL = TRANS_MIL(in_dim=in_dim, num_classes=num_classes, dropout=0.1,act=nn.ReLU())
            self.agg_feature_dim = 512
        elif MIL_Name == 'WIKG_MIL':
            from modules.WIKG_MIL.wikg_mil import WIKG_MIL
            self.MIL = WIKG_MIL(in_dim=in_dim,num_classes=num_classes,dropout=0.1,dim_hidden=512)
            self.agg_feature_dim = 512


        self.classifier_c = nn.Linear(self.agg_feature_dim, num_classes)        
        self.classifier_r = nn.Linear(self.agg_feature_dim, num_classes)        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_c, x_r=None, alpha=None,**kwargs):

        if kwargs.get('return_WSI_feature') or kwargs.get('return_WSI_attn'):
            return self.MIL(x_c, **kwargs)
        forward_return = {}
        feat_c = self.MIL(x_c, return_WSI_feature=True)['WSI_feature']
        

        if self.training and x_r is not None and alpha is not None:
            feat_r = self.MIL(x_r, return_WSI_feature=True)['WSI_feature']
            
            logits_c = self.classifier_c(feat_c) # W_c * f_c
            logits_r = self.classifier_r(feat_r) # W_r * f_r
            logits_mixed = alpha * logits_c + (1 - alpha) * logits_r
            forward_return['logits'] = logits_mixed
            return forward_return
        else:
            logits_c = self.classifier_c(feat_c)
            logits_r = self.classifier_r(feat_c) 
            
            logits_final = 0.5 * logits_c + 0.5 * logits_r
            forward_return['logits'] = logits_final
            return forward_return