import torch
from torch.utils.data import DataLoader
from modules.TRANS_MIL.trans_mil import TRANS_MIL
from utils.process_utils import get_process_pipeline,get_act
from utils.wsi_utils import WSI_Dataset
from utils.general_utils import set_global_seed,init_epoch_info_log,add_epoch_info_log,early_stop
from utils.model_utils import get_optimizer,get_scheduler,get_criterion,save_last_model,save_log,model_select
from utils.loop_utils import train_loop,val_loop,bbn_train_loop
from utils.lt_utils import apply_tnorm_temporary, restore_tnorm, setup_crt, setup_lws
from tqdm import tqdm
    
from models.BBN import BBN_Adapter

def process_TRANS_MIL(args):
    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'train')
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'val')
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'test')
    process_pipeline = get_process_pipeline(val_dataset,test_dataset) 
    args.General.process_pipeline = process_pipeline
    '''
    generator settings
    '''
    
    generator = torch.Generator()
    generator.manual_seed(args.General.seed) 
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    # use_balanced_sampler = args.Dataset.balanced_sampler.use
    # if use_balanced_sampler:
    #     sampler = train_dataset.get_balanced_sampler(replacement = args.Dataset.balanced_sampler.replacement)
    #     train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers = num_workers,generator=generator,sampler=sampler)
    # else:
    #     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = num_workers,generator=generator)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    if args.LT.method == "Resampling":
        train_sampler = train_dataset.get_balanced_sampler()
        train_shuffle = False
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=train_shuffle, num_workers = num_workers,generator=generator,sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    elif args.LT.method == "BBN":
        train_loader_c = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        balanced_sampler = train_dataset.get_balanced_sampler()
        train_loader_r = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers, generator=generator, sampler=balanced_sampler)        
    else:
        train_sampler = None
        train_shuffle = True
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=train_shuffle, num_workers = num_workers,generator=generator,sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)    

    print('DataLoader Ready!')
    
    device = torch.device(f'cuda:{args.General.device}')
    num_classes = args.General.num_classes
    in_dim = args.Model.in_dim
    dropout = args.Model.dropout
    act = get_act(args.Model.act)
    if args.LT.method == 'BBN':
        mil_model= BBN_Adapter(in_dim=in_dim, num_classes=num_classes,MIL_Name='TRANS_MIL')
    else:
        mil_model = TRANS_MIL(num_classes=num_classes,dropout=dropout,act=act,in_dim=in_dim)
    #mil_model = TRANS_MIL(num_classes=num_classes,dropout=dropout,act=act,in_dim=in_dim)
    mil_model.to(device)
    
    print('Model Ready!')
    
    optimizer,base_lr = get_optimizer(args,mil_model)
    scheduler,warmup_scheduler = get_scheduler(args,optimizer,base_lr)
    criterion = get_criterion(args,cls_num_list=train_dataset.cls_num_list)
    if hasattr(criterion, 'to'):
        criterion.to(device)
    #criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup
    
    '''
    begin training
    '''
    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    REVERSE = False
    best_val_metric = 0
    if best_model_metric == 'val_loss':
        REVERSE = True
        best_val_metric = 9999
    best_epoch = 1

    crt_stage_len = int(args.General.num_epochs * 0.2) 
    crt_start_epoch = args.General.num_epochs - crt_stage_len
    is_crt_phase = False

    print('Start Process!')
    print('Using Process Pipeline:',process_pipeline)
    for epoch in tqdm(range(args.General.num_epochs),colour='GREEN'):
        if args.LT.method == 'cRT' and epoch == crt_start_epoch and not is_crt_phase:
            print(f"\n[cRT] Reached epoch {epoch}. Switching to Stage 2 (Classifier Re-Training)...")
            

            mil_model = setup_crt(mil_model)

            crt_params = [p for p in mil_model.parameters() if p.requires_grad]
            
            optimizer = torch.optim.Adam(crt_params, lr=base_lr*0.1, weight_decay=0.00001)

            print("🔄 [cRT] Switching DataLoader to Class-Balanced Sampler...")
            
            balanced_sampler = train_dataset.get_balanced_sampler()
            
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=num_workers, 
                generator=generator, 
                sampler=balanced_sampler
            )
            

            now_scheduler = None 

            is_crt_phase = True
            print("[cRT] Stage 2 Started: Backbone Frozen, Head Reset, Optimizer Replaced.")
        elif args.LT.method == 'LWS' and epoch == crt_start_epoch and not is_crt_phase:
            print(f"\n[LWS] Reached epoch {epoch}. Switching to Stage 2...")
                
            mil_model = setup_lws(mil_model, num_classes)
            mil_model.to(device) 
            
            train_params = [p for p in mil_model.parameters() if p.requires_grad]
            
            optimizer = torch.optim.Adam(train_params, lr=base_lr*0.1, weight_decay=0)
            
            print("🔄 [LWS] Switching to Class-Balanced Sampler...")
            balanced_sampler = train_dataset.get_balanced_sampler()
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers, generator=generator, sampler=balanced_sampler)

            now_scheduler = None
            is_crt_phase = True
            print("[LWS] Stage 2 Started: Only learning scales.")
    

        if not is_crt_phase:
            if epoch+1 <= warmup_epoch:
                now_scheduler = warmup_scheduler
            else:
                now_scheduler = scheduler
            
        # if epoch+1 <= warmup_epoch:
        #     now_scheduler = warmup_scheduler
        # else:
        #     now_scheduler = scheduler
        if args.LT.method == 'BBN':
            train_loss, alpha_val, cost_time = bbn_train_loop(epoch, args.General.num_epochs, device, mil_model, train_loader_c, train_loader_r, criterion, optimizer, now_scheduler)
        else:
            train_loss,cost_time = train_loop(device,mil_model,train_dataloader,criterion,optimizer,now_scheduler,epoch)

        #train_loss,cost_time = train_loop(device,mil_model,train_dataloader,criterion,optimizer,now_scheduler)
        original_weights_backup = None
        if args.LT.method == 'T-Norm':
            original_weights_backup = apply_tnorm_temporary(mil_model, tau=1.0)
        
        
        if process_pipeline == 'Train_Val_Test':
            val_loss,val_metrics = val_loop(device,num_classes,mil_model,val_dataloader,criterion,cls_num_list=train_dataset.cls_num_list)
            test_loss,test_metrics = val_loop(device,num_classes,mil_model,test_dataloader,criterion,cls_num_list=train_dataset.cls_num_list)
        elif process_pipeline == 'Train_Val':
            val_loss,val_metrics = val_loop(device,num_classes,mil_model,val_dataloader,criterion,cls_num_list=train_dataset.cls_num_list)
            test_loss,test_metrics = None,None
        elif process_pipeline == 'Train_Test':
            val_loss,val_metrics,test_loss,test_metrics = None,None,None,None
            if epoch+1 == args.General.num_epochs:
                test_loss,test_metrics = val_loop(device,num_classes,mil_model,test_dataloader,criterion,cls_num_list=train_dataset.cls_num_list)
       
        if args.LT.method == 'T-Norm':
            restore_tnorm(mil_model, original_weights_backup)

        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}EPOCH:{ENDC}{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}\n')
        print(f'{FAIL}Val_Metrics:  {ENDC}{val_metrics}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics)
        
        # model selection, it only works when process_pipeline is 'Train_Val_Test' or 'Train_Val'
        best_val_metric,best_epoch = model_select(REVERSE,args,mil_model.state_dict(),val_metrics,best_model_metric,best_val_metric,epoch,best_epoch)

        '''
        early stop
        '''
        if early_stop(args,epoch_info_log,process_pipeline,epoch,mil_model.state_dict(),best_epoch):
            break

        if epoch+1 == args.General.num_epochs:
            save_last_model(args,mil_model.state_dict(),epoch+1)
            save_log(args,epoch_info_log,best_epoch,process_pipeline)





