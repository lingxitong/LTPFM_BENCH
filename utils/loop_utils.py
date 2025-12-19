import torch
import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import time
import inspect
import torch.nn as nn

def _calc_ece(probs, labels, n_bins=15):
    if probs.shape[0] == 0: return 0.0
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def cal_scores(probs, labels, num_classes,cls_num_list=None):      
    probs = torch.tensor(probs)
    labels = torch.tensor(labels)
    predicted_classes = torch.argmax(probs, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    if num_classes > 2:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='micro', multi_class='ovr')
        weighted_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='weighted', multi_class='ovr')
    else:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())
        weighted_auc = micro_auc = macro_auc
    weighted_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    weighted_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    weighted_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    macro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    macro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    macro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    micro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    micro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    micro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='micro') 
    baccuracy = balanced_accuracy_score(labels.numpy(), predicted_classes.numpy())
    quadratic_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')
    linear_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='linear')
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())

    tail_ece = -1.0 # 默认值
    
    if cls_num_list is not None:
        labels_for_tail = labels.view(-1)
        TAIL_THRESH = 20
        tail_indices = [i for i, c in enumerate(cls_num_list) if c < TAIL_THRESH]
        
        if len(tail_indices) == 0:
            sorted_indices = np.argsort(cls_num_list) 
            num_tail = int(len(cls_num_list) / 3)
            if num_tail == 0 and len(cls_num_list) > 0:
                num_tail = 1
                
            tail_indices = sorted_indices[:num_tail].tolist()
            # print(f"DEBUG: Adaptive Tail Indices: {tail_indices}") # 调试用

        if len(tail_indices) > 0:
            is_tail = torch.zeros_like(labels_for_tail, dtype=torch.bool)
            for idx in tail_indices:
                is_tail |= (labels_for_tail == idx)
            
            tail_probs = probs[is_tail]
            tail_labels = labels_for_tail[is_tail]
            
            if tail_probs.shape[0] > 0:
                tail_ece = _calc_ece(tail_probs, tail_labels)
            else:
                tail_ece = 0.0 
        else:
            tail_ece = 0.0 

    metrics = {'acc': accuracy,  'bacc': baccuracy, 
               'macro_auc': macro_auc, 'micro_auc': micro_auc, 'weighted_auc':weighted_auc,
                'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1, 
                 'macro_recall': macro_recall, 'micro_recall': micro_recall,'weighted_recall': weighted_recall, 
                 'macro_pre': macro_precision, 'micro_pre': micro_precision,'weighted_pre': weighted_precision,
                 'quadratic_kappa': quadratic_kappa,'linear_kappa':linear_kappa,  
                 'tail_ece': tail_ece,
                 'confusion_mat': confusion_mat}
    return metrics

def train_loop(device,model,loader,criterion,optimizer,scheduler,epoch):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    forward_parms = inspect.signature(criterion.forward).parameters
    needs_epoch = 'epoch' in forward_parms
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        train_logits = model(bag)['logits']
        if needs_epoch:
            train_loss = criterion(train_logits, label, epoch)
        else:
            train_loss = criterion(train_logits, label)
        #train_loss = criterion(train_logits, label)
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def bbn_train_loop(epoch, max_epochs, device, model, loader_c, loader_r, criterion, optimizer,scheduler):
    model.train()
    start = time.time()
    div_epoch = epoch / max_epochs
    alpha = 1 - (div_epoch ** 2)
    
    total_loss = 0.
    
    for i, (data_c, data_r) in enumerate(zip(loader_c, loader_r)):
        optimizer.zero_grad()
        
        # data_c: 来自 Uniform Loader
        # data_r: 来自 Balanced Loader
        bag_c, label_c = data_c[0].to(device).float(), data_c[1].long().to(device)
        bag_r, label_r = data_r[0].to(device).float(), data_r[1].long().to(device)

        output = model(bag_c, x_r=bag_r, alpha=alpha)
        logits_mixed = output['logits']
        loss = alpha * criterion(logits_mixed, label_c) + \
               (1 - alpha) * criterion(logits_mixed, label_r)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()    
    avg_loss = total_loss / (i + 1)
    end = time.time()
    total_time = end - start
    return avg_loss, alpha, total_time

#Using the standard CrossEntropyLoss
def val_loop(device,num_classes,model,loader,criterion,cls_num_list=None,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            if retrun_WSI_feature:
                WSI_feature = model(bag,return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            val_logits = model(bag)['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            
            val_loss = F.cross_entropy(val_logits, label)
            val_loss_log += val_loss.item()
            # Handle BCE loss - convert label to one-hot if needed
            # if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            #     import torch.nn.functional as F
            #     label_for_loss = F.one_hot(label.long(), num_classes=num_classes).float()
            # else:
            #     label_for_loss = label
            
            # val_loss = criterion(val_logits, label_for_loss)
            # val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes,cls_num_list=cls_num_list)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics

def ac_train_loop(device,model,loader,criterion,optimizer,scheduler,n_token):
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag)
        train_logits = forward_return['logits']
        sub_preds = forward_return['sub_preds']
        attns = forward_return['attns']
        if n_token > 1:
            loss0 = criterion(sub_preds, label.repeat_interleave(n_token))
        else:
            loss0 = torch.tensor(0.)
        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        attns = torch.softmax(attns, dim=-1)

        for i in range(n_token):
            for j in range(i + 1, n_token): 
                diff_loss += torch.cosine_similarity(attns[:, i], attns[:, j], dim=-1).mean() / (
                            n_token * (n_token - 1) / 2)
        train_loss = criterion(train_logits, label)
        train_loss = diff_loss + loss0 + train_loss
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def ac_val_loop(device,num_classes,model,loader,criterion,n_token,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].long().to(device)
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            forward_return = model(bag)
            val_logits = forward_return['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            sub_preds = forward_return['sub_preds']
            attns = forward_return['attns']
            if n_token > 1:
                loss0 = criterion(sub_preds, label.repeat_interleave(n_token))
            else:
                loss0 = torch.tensor(0.)
            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attns = torch.softmax(attns, dim=-1)
            for i in range(n_token):
                for j in range(i + 1, n_token): 
                    diff_loss += torch.cosine_similarity(attns[:, i], attns[:, j], dim=-1).mean() / (
                                n_token * (n_token - 1) / 2)
            val_loss = criterion(val_logits, label)
            val_loss = diff_loss + loss0 + val_loss
            val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics


# clam has instance-loss defferent from other mil models
def clam_train_loop(device,model,loader,criterion,optimizer,scheduler,bag_weight,epoch):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    forward_parms = inspect.signature(criterion.forward).parameters
    needs_epoch = 'epoch' in forward_parms
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag,label=label)
        instance_loss = forward_return['instance_loss']
        train_logits = forward_return['logits']
        if needs_epoch:
            train_loss = criterion(train_logits, label, epoch)
        else:
            train_loss = criterion(train_logits, label)
        #train_loss = criterion(train_logits, label)
        total_loss = train_loss * bag_weight + instance_loss * (1-bag_weight)
        train_loss_log += total_loss.item()
        total_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def tripleloss(golabal,p_center,nc_center):
    golabal = golabal.squeeze(0)
    n_globallesionrepresente, _ = golabal.shape
    p_center = p_center.repeat(n_globallesionrepresente, 1)
    nc_center = nc_center.repeat(n_globallesionrepresente, 1)
    triple_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y) ,margin=1)
    loss = triple_loss(golabal,p_center,nc_center)
    return loss

def dgr_train_loop(device,model,loader,criterion,optimizer,scheduler,now_epoch,epoch_des,n_lesion):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        if torch.argmax(label)==0:
            forward_return = model(bag,bag_mode='normal')
            train_logits, A,H,p_center,nc_center,lesion = forward_return['logits'],forward_return['A'],forward_return['H'],forward_return['postivecenter'],forward_return['normalcenter'],forward_return['lesion_enhacing']
        else:
            train_logits, A,H,p_center,nc_center,lesion= model(bag,bag_mode='abnormal')
        if now_epoch < epoch_des:
            train_loss = criterion(train_logits, label)
        else:
            train_loss = criterion(train_logits, label)
            lesion_norm = lesion.squeeze(0)
            lesion_norm = torch.nn.functional.normalize(lesion_norm)
            div_loss = -torch.logdet(lesion_norm@lesion_norm.T+1e-10*torch.eye(n_lesion).to(device))
            sim_loss = tripleloss(lesion,p_center,nc_center)
            train_loss = train_loss + 0.1*div_loss + 0.1*sim_loss 

        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time

def clam_val_loop(device,num_classes,model,loader,criterion,bag_weight,cls_num_list=None,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            if retrun_WSI_feature:
                WSI_feature = model(bag,label = label, return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,label = label, return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            forward_return = model(bag,label=label)
            instance_loss = forward_return['instance_loss']
            val_logits = forward_return['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            #val_loss = criterion(val_logits,label)
            val_loss = F.cross_entropy(val_logits, label)
            total_loss = val_loss * bag_weight + instance_loss * (1-bag_weight)
            val_loss_log += total_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes,cls_num_list=cls_num_list)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics

def ds_train_loop(device,model,loader,criterion,optimizer,scheduler):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    model = model.to(device)
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag)
        max_prediction = forward_return['max_prediction']
        train_logits = forward_return['logits']
        loss_bag = criterion(train_logits, label)
        loss_max = criterion(max_prediction, label)
        train_loss = 0.5*loss_bag + 0.5*loss_max
        train_loss_log += train_loss.item()

        train_loss.backward()

        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time

def ds_val_loop(device,num_classes,model,loader,criterion,retrun_WSI_feature = False,return_WSI_attn=False):
    WSI_features = []
    WSI_attns = []
    labels = []
    bag_predictions_after_normal = []
    val_loss_log = 0
    model.eval()
    model = model.to(device)
    with torch.autograd.set_detect_anomaly(True):
        for i, data in enumerate(loader):
            label = data[1].long().to(device)
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            forward_return = model(bag)
            if retrun_WSI_feature:
                WSI_feature = model(bag,return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            max_prediction = forward_return['max_prediction']
            val_logits = forward_return['logits']
            bag_predictions_after_normal.append(torch.softmax(val_logits[0],0).cpu().detach().numpy())
            loss_bag = criterion(val_logits, label)
            loss_max = criterion(max_prediction, label)
            val_loss = 0.5*loss_bag + 0.5*loss_max
            val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().detach().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_loss_log /= len(loader)
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    return val_loss_log,val_metrics

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def dtfd_train_loop(device, model_list, loader, criterion, optimizer_list, scheduler_list, num_Group, grad_clipping,distill,total_instance):
    train_loss_log = 0
    start = time.time()
    instance_per_group = total_instance // num_Group
    # Unpack model list
    classifier, attention, dimReduction, attCls = model_list
    classifier.train()
    attention.train()
    dimReduction.train()
    attCls.train()

    # Unpack optimizer and scheduler lists
    optimizer_A, optimizer_B = optimizer_list
    scheduler_A, scheduler_B = scheduler_list

    total_loss = 0
    for i, data in enumerate(loader):
        label = data[1].long().to(device)
        bag = data[0].to(device).float()

        slide_sub_preds = []
        slide_sub_labels = []
        slide_pseudo_feat = []

        # Split bag into chunks
        inputs_pseudo_bags = torch.chunk(bag.squeeze(0), num_Group, dim=0)

        for subFeat_tensor in inputs_pseudo_bags:
            slide_sub_labels.append(label)
            subFeat_tensor = subFeat_tensor.to(device)

            # Forward pass through models
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
            tPredict = classifier(tattFeat_tensor)  # 1 x 2
            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)
            slide_sub_preds.append(tPredict)


        # Concatenate tensors
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  # numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  # numGroup

        # Calculate and backpropagate loss for the first tier
        loss_A = criterion(slide_sub_preds, slide_sub_labels)
        optimizer_A.zero_grad()
        loss_A.backward(retain_graph=True)
        total_loss += loss_A.item()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clipping)


        # Second tier optimization
        gSlidePred = attCls(slide_pseudo_feat)['logits']
        loss_B = criterion(gSlidePred, label).mean()
        optimizer_B.zero_grad()
        loss_B.backward()
        total_loss += loss_B.item()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(attCls.parameters(), grad_clipping)
        optimizer_A.step()
        optimizer_B.step()

    # Step schedulers
    scheduler_A.step()
    scheduler_B.step()

    end = time.time()
    total_loss /= len(loader)
    total_time = end - start

    return total_loss, total_time


def dtfd_val_loop(device,num_classes,model_list,loader,criterion,num_Group,grad_clipping,distill,total_instance,retrun_WSI_feature = False,return_WSI_attn=False):
    WSI_features = []
    WSI_attns = []
    instance_per_group = total_instance // num_Group
    classifier,attention,dimReduction,attCls = model_list
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    attCls.eval()
    total_loss = 0
    y_score=[]
    y_true=[]
    for i, data in enumerate(loader):
        label = data[1].long().to(device)
        bag = data[0].to(device).float()

        slide_sub_preds = []
        slide_sub_labels = []
        slide_pseudo_feat = []
        inputs_pseudo_bags=torch.chunk(bag.squeeze(0), num_Group,dim=0)
        
        for subFeat_tensor in inputs_pseudo_bags:
            subFeat_tensor=subFeat_tensor.to(device)
            with torch.no_grad():
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
                tPredict = classifier(tattFeat_tensor)  # 1 x 2
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)
            slide_sub_preds.append(tPredict)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        gSlidePred = torch.softmax(attCls(slide_pseudo_feat)['logits'], dim=1)
        forward_return = attCls(slide_pseudo_feat, return_WSI_attn = return_WSI_attn, return_WSI_feature = retrun_WSI_feature)
        if retrun_WSI_feature:
            WSI_feature = forward_return['WSI_feature']
            WSI_features.append(WSI_feature)
            continue
        if return_WSI_attn:
            WSI_attn = forward_return['WSI_attn']
            WSI_attns.append(WSI_attn)
            continue
        loss = criterion(forward_return['logits'], label)
        total_loss += loss.item()
        pred=(gSlidePred.cpu().data.numpy()).tolist()
        y_score.extend(pred)
        y_true.extend(label)
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().detach().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    
    total_loss /= len(loader)
    val_metrics= cal_scores(y_score,y_true,num_classes)
    return total_loss,val_metrics


# ============================================
# Mixup Training Loop Utilities
# ============================================
import numpy as np

def train_loop_with_mixup(device, model, dataloader, criterion, optimizer, scheduler, mixup_config, mix_fn):
    """
    Generic training loop with mixup augmentation
    Args:
        device: torch device
        model: MIL model
        dataloader: training dataloader
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler
        mixup_config: dict with mixup parameters (prob, alpha, etc.)
        mix_fn: mixing function from the specific module (e.g., mixup_data, insmix_data, etc.)
    Returns:
        train_loss, cost_time
    """
    model.train()
    train_loss = 0.
    start_time = time.time()
    
    all_features = []
    all_labels = []
    
    for idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            _, data, label = batch
        else:
            data, label = batch
        all_features.append(data.squeeze(0).to(device))
        all_labels.append(label.to(device))
    
    prob = mixup_config.get('prob', 0.5)
    n_samples = len(all_features)
    perm = torch.randperm(n_samples)
    
    for i in range(n_samples):
        optimizer.zero_grad()
        
        if np.random.rand() < prob:
            j = perm[i].item()
            if i != j:
                # Call the specific mix function with appropriate kwargs
                mix_kwargs = {k: v for k, v in mixup_config.items() if k != 'prob'}
                # For rankmix, we need to pass the model
                if 'model' in mix_fn.__code__.co_varnames:
                    mixed_feat, mixed_label, _ = mix_fn(
                        all_features[i], all_labels[i],
                        all_features[j], all_labels[j],
                        model, **mix_kwargs
                    )
                else:
                    mixed_feat, mixed_label, _ = mix_fn(
                        all_features[i], all_labels[i],
                        all_features[j], all_labels[j],
                        **mix_kwargs
                    )
            else:
                mixed_feat, mixed_label = all_features[i], all_labels[i]
        else:
            mixed_feat, mixed_label = all_features[i], all_labels[i]
        
        forward_return = model(mixed_feat.unsqueeze(0))
        logits = forward_return['logits']
        
        if mixed_label.dim() == 0 or mixed_label.size(0) == 1:
            if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                label_for_loss = F.one_hot(mixed_label.long(), num_classes=logits.size(1)).float()
            else:
                label_for_loss = mixed_label.long()
        else:
            label_for_loss = mixed_label.float()
        
        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            loss = criterion(logits, label_for_loss)
        else:
            loss = criterion(logits, label_for_loss.argmax(dim=-1) if label_for_loss.dim() > 1 else label_for_loss)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()
    
    train_loss /= n_samples
    cost_time = time.time() - start_time
    
    return train_loss, cost_time
