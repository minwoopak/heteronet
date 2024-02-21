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


# import torch
# from scipy.stats import pearsonr,spearmanr

# def average_pcc(tensor1, tensor2):
#     """
#     Calculates the average PCC between the rows of two 2D tensors.
#     """
#     assert tensor1.shape[0] == tensor2.shape[0], "Tensors must have the same number of rows."
    
#     pccs = []
#     for i in range(tensor1.shape[0]):
#         row1 = tensor1[i]
#         row2 = tensor2[i]
#         pcc, _ = pearsonr(row1, row2)
#         pccs.append(pcc)
    
#     avg_pcc = torch.tensor(pccs).mean()
#     return avg_pcc

# def average_spearman(tensor1, tensor2):
#     """
#     Calculates the average Spearman correlation between the rows of two 2D tensors.
#     """
#     assert tensor1.shape[0] == tensor2.shape[0], "Tensors must have the same number of rows."
    
#     spearman_corrs = []
#     for i in range(tensor1.shape[0]):
#         row1 = tensor1[i]
#         row2 = tensor2[i]
#         spearman_corr, _ = spearmanr(row1, row2)
#         spearman_corrs.append(spearman_corr)
    
#     avg_spearman = torch.tensor(spearman_corrs).mean()
#     return avg_spearman