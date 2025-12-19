import torch
import torch.nn as nn
import inspect
def _get_target_layers(model):

    target_attr_names = ['classifier', 'fc', 'head', '_fc2', 'classifiers']
    
    found_layers = []

    for name in target_attr_names:
        if hasattr(model, name):
            target = getattr(model, name)
            if isinstance(target, nn.ModuleList):
                for sub_layer in target:
                    if isinstance(sub_layer, nn.Linear):
                        found_layers.append(sub_layer)
                if found_layers: return found_layers
            elif isinstance(target, nn.Sequential):
                for sub_layer in reversed(target):
                    if isinstance(sub_layer, nn.Linear):
                        found_layers.append(sub_layer)
                        return found_layers 
            elif isinstance(target, nn.Linear):
                found_layers.append(target)
                return found_layers

    return None

def apply_tnorm_temporary(model, tau=1.0):

    target_layers = _get_target_layers(model)
        
    if not target_layers:
        print("⚠️ Warning: T-Norm could not find any Linear layer. Checked: classifier, fc, head, _fc2, classifiers")
        return None

    original_weights_backup = {}

    for layer in target_layers:
        original_weights_backup[layer] = layer.weight.data.clone()
        weights = layer.weight.data
        norms = torch.norm(weights, p=2, dim=1, keepdim=True)
        weights_new = weights / (norms.pow(tau) + 1e-12)
        layer.weight.data = weights_new
    
    # print(f"🚀 T-Norm activated on {len(target_layers)} layers.")
    return original_weights_backup

def restore_tnorm(model, original_weights_backup):

    if original_weights_backup is None:
        return

    for layer, original_data in original_weights_backup.items():
        layer.weight.data = original_data
    
    # print("🔄 T-Norm deactivated.")

# crt
def setup_crt(model):
    
    print("🔧 [cRT] Setting up Classifier Re-Training (Stage 2)...")
    
    for param in model.parameters():
        param.requires_grad = False

    target_layers = _get_target_layers(model)
    
    if not target_layers:
        print("❌ Error: cRT could not find classifier layers.")
        return model
    
    for layer in target_layers:
        for param in layer.parameters():
            param.requires_grad = True
            

        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
                
    print(f"✅ [cRT] Backbone frozen. {len(target_layers)} classifier heads reset & unfrozen.")
    return model

#LWS
class LWS_Model_Wrapper(nn.Module):

    def __init__(self, model, num_classes):
        super(LWS_Model_Wrapper, self).__init__()
        
        self.backbone_model = model
        
        for param in self.backbone_model.parameters():
            param.requires_grad = False
            
        self.scales = nn.Parameter(torch.ones(1, num_classes))
        forward_params = inspect.signature(self.backbone_model.forward).parameters
        self.accepts_label = 'label' in forward_params
        print(f"🔧 [LWS Wrapper] Original model frozen. Initialized learnable scales for {num_classes} classes.")

    def forward(self, x, label=None,**kwargs):
        if self.accepts_label:
            outputs = self.backbone_model(x, label=label, **kwargs)
        else:
            outputs = self.backbone_model(x, **kwargs)

        original_logits = outputs['logits']        
        new_logits = original_logits * self.scales
        outputs['logits'] = new_logits
        return outputs

def setup_lws(model, num_classes):
    lws_model = LWS_Model_Wrapper(model, num_classes)
    return lws_model