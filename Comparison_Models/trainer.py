import torch

def train(model, epoch, train_loader, optimizer, loss_fn, device, display_step=100):
    # ====== Train ====== #
    list_train_batch_loss = []
    list_train_batch_out  = []
    list_train_batch_true = []
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = tuple(d.to(device) for d in data)
        true_y = target.to(device)
        
        optimizer.zero_grad()
        pred_y = model(data)
        loss = loss_fn(pred_y, true_y)
        
        loss.backward()
        optimizer.step()
        
        list_train_batch_out.extend(pred_y.detach().cpu().numpy())
        list_train_batch_true.extend(true_y.detach().cpu().numpy())
        
        list_train_batch_loss.append(loss.detach().cpu().numpy())
    
        if batch_idx % display_step == 0 and batch_idx !=0:
            print(f'Epoch: {epoch}, minibatch: {batch_idx}, TRAIN: loss: {loss:.4f}')
        
    return model, list_train_batch_loss, list_train_batch_out, list_train_batch_true


def validate(model, valid_loader, loss_fn, device):
    list_val_batch_loss = []
    list_val_batch_out  = []
    list_val_batch_true = []
    
    model.eval()
    for batch_idx, (data, target) in enumerate(valid_loader):
        data = tuple(d.to(device) for d in data)
        true_y = target.to(device)
        
        pred_y = model(data)
        loss = loss_fn(pred_y, true_y)
        
        list_val_batch_out.extend(pred_y.detach().cpu().numpy())
        list_val_batch_true.extend(true_y.detach().cpu().numpy())

        list_val_batch_loss.append(loss.detach().cpu().numpy())
    
    return list_val_batch_loss, list_val_batch_out, list_val_batch_true


def test(model, test_loader, loss_fn, device):
    with torch.no_grad():
        # ====== Test ====== #
        list_test_loss = []
        list_test_out  = []
        list_test_true = []

        model.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            data = tuple(d.to(device) for d in data)
            true_y = target.to(device)
            
            pred_y = model(data)
            loss = loss_fn(pred_y, true_y)
            
            list_test_out.extend(pred_y.detach().cpu().numpy())
            list_test_true.extend(true_y.detach().cpu().numpy())

            list_test_loss.append(loss.detach().cpu().numpy())
    
    #test_loss = sum(list_test_loss)/len(list_test_loss)
        
    #print(f"Test: Loss: {test_loss:.4f}, RMSE; {test_rmse:.4f}, CORR: {test_corr:.4f}, SPEARMAN: {test_spearman:.4f}")
    return list_test_loss, list_test_out, list_test_true


# =============================== #
# ====== Define Experiment ====== #
# =============================== #
import os
from utils import logging
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
from copy import deepcopy
import json
import torch.optim as optim
import numpy as np

def experiment(name_var1, name_var2, var1, var2, args, dataset_partition, model, loss_fn, device):
    
    # === Optimizer === #
    optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': loss_fn.parameters()}
        ], lr=args.learning_rate, weight_decay=args.weight_decay)

    # === Scheduler === #
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    # scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1, verbose=True)
    
    # ====== Cross Validation Best Performance Dict ====== #
    best_performances = {}
    best_performances['best_epoch'] = 0
    best_performances['best_train_loss'] = float('inf')
    best_performances['best_train_corr'] = 0.0
    best_performances['best_valid_loss'] = float('inf')
    best_performances['best_valid_corr'] = 0.0
    # ==================================================== #
    
    list_epoch = []
    list_train_epoch_loss = []
    list_epoch_rmse = []
    list_epoch_corr = []
    list_epoch_spearman = []

    list_val_epoch_loss = []
    list_val_epoch_rmse = []
    list_val_epoch_corr = []
    list_val_spearman = []
    
    counter = 0
    for epoch in range(args.epochs):
        list_epoch.append(epoch)
        
        # ====== TRAIN Epoch ====== #
        model, list_train_batch_loss, list_train_batch_out, list_train_batch_true = train(model, epoch, dataset_partition['train_loader'], optimizer, loss_fn, device, args.display_step)
        list_train_batch_out  = torch.from_numpy(np.array(list_train_batch_out))
        list_train_batch_true = torch.from_numpy(np.array(list_train_batch_true))
        
        # === Calculate Performance Metrics === #
        epoch_train_rmse = np.sqrt(F.mse_loss(list_train_batch_out, list_train_batch_true)).item()
        try:
            epoch_train_corr, _ = pearsonr(list_train_batch_out.squeeze(), list_train_batch_true.squeeze())
            epoch_train_spearman, _ = spearmanr(list_train_batch_out.squeeze(), list_train_batch_true.squeeze())
        except:
            print(list_train_batch_out.shape, list_train_batch_true.shape, list_train_batch_out, list_train_batch_true)
            epoch_train_corr = 0.0
            epoch_train_spearman = 0.0
                
        train_epoch_loss = sum(list_train_batch_loss) / len(list_train_batch_loss)
        list_train_epoch_loss.append(train_epoch_loss)
        list_epoch_rmse.append(epoch_train_rmse)
        list_epoch_corr.append(epoch_train_corr)
        list_epoch_spearman.append(epoch_train_spearman)
        
        # ====== VALID Epoch ====== #
        list_val_batch_loss, list_val_batch_out, list_val_batch_true = validate(model, dataset_partition['valid_loader'], loss_fn, device)
        list_val_batch_out  = torch.from_numpy(np.array(list_val_batch_out))
        list_val_batch_true = torch.from_numpy(np.array(list_val_batch_true))
        
        # === Calculate Performance Metrics === #
        epoch_val_rmse = np.sqrt(F.mse_loss(list_val_batch_out, list_val_batch_true)).item()
        epoch_val_corr, _ = pearsonr(list_val_batch_out.squeeze(), list_val_batch_true.squeeze())
        epoch_val_spearman, _ = spearmanr(list_val_batch_out.squeeze(), list_val_batch_true.squeeze())
        
        val_epoch_loss = sum(list_val_batch_loss)/len(list_val_batch_loss)
        list_val_epoch_loss.append(val_epoch_loss)
        list_val_epoch_rmse.append(epoch_val_rmse)
        list_val_epoch_corr.append(epoch_val_corr)
        list_val_spearman.append(epoch_val_spearman)
        
        if val_epoch_loss < best_performances['best_valid_loss']:
            best_performances['best_epoch'] = epoch
            best_performances['best_train_loss'] = train_epoch_loss
            best_performances['best_train_corr'] = epoch_train_corr
            best_performances['best_valid_loss'] = val_epoch_loss
            best_performances['best_valid_corr'] = epoch_val_corr
            torch.save(model, os.path.join(args.outdir, args.exp_name + f'_{name_var1}{var1}_{name_var2}{var2}.model'))
            model_max = deepcopy(model)
            
            counter = 0
        else:
            counter += 1
            logging(f'Early Stopping counter: {counter} out of {args.patience}', args.outdir, args.exp_name+'.log')
        
        logging(f'Epoch: {epoch:02d}, Train loss: {list_train_epoch_loss[-1]:.4f}, rmse: {epoch_train_rmse:.4f}, corr: {epoch_train_corr:.4f}, Valid loss: {list_val_epoch_loss[-1]:.4f}, rmse: {epoch_val_rmse:.4f}, pcc: {epoch_val_corr:.4f}', args.outdir, args.exp_name+'.log')
        if counter == args.patience:
            break
        
        scheduler.step(list_val_epoch_loss[-1])
        
    if args.testset_yes:
        list_test_loss, list_test_out, list_test_true = test(model_max, dataset_partition['test_loader'], loss_fn, device)
        list_test_out  = torch.from_numpy(np.array(list_test_out))
        list_test_true = torch.from_numpy(np.array(list_test_true))
        
        test_rmse = np.sqrt(F.mse_loss(list_test_out, list_test_true)).item()
        test_corr, _ = pearsonr(list_test_out.squeeze(), list_test_true.squeeze())
        test_spearman, _ = spearmanr(list_test_out.squeeze(), list_test_true.squeeze())
        
        test_loss = sum(list_test_loss)/len(list_test_loss)

        logging(f"Test:\tLoss: {test_loss}\tRMSE: {test_rmse}\tCORR: {test_corr}\tSPEARMAN: {test_spearman}", args.outdir, f'{args.exp_name}_test.log')
        
        # test_log_df = pd.DataFrame({
        #     'test_loss': list_test_loss,
        #     'test_out': list_test_out,
        #     'test_true': list_test_true,
        # })
        # test_log_df = test_log_df.sort_values(by='test_loss', ascending=True)
        # filename = os.path.join(args.outdir, f'{args.exp_name}_{name_var1}{var1}_{name_var2}{var2}_test.result')
        # test_log_df.to_csv(filename, sep='\t', header=True, index=False)
    
    # ====== Add Result to Dictionary ====== #
    result = {}
    result['train_losses'] = list_train_epoch_loss
    result['val_losses'] = list_val_epoch_loss
    result['train_accs'] = list_epoch_corr
    result['val_accs'] = list_val_epoch_corr
    result['train_acc'] = epoch_train_corr
    result['val_acc'] = epoch_val_corr
    if args.testset_yes:
        result['test_acc'] = test_corr
    
    filename = os.path.join(args.outdir, f'{args.exp_name}_{name_var1}{var1}_{name_var2}{var2}_best_performances.json')
    with open(filename, 'w') as f:
        json.dump(best_performances, f)
        
    return vars(args), result, best_performances, model_max
