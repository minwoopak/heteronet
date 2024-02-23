# === Fixed === #
import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
import random
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from collections import defaultdict

# === Task-specific === #
from dataset import DrugResponse_Dataset
from torch_geometric.loader import DataLoader
from model import HeteroNet_EmbConcatTotal
from utils import save_exp_result, logging

from sklearn.model_selection import train_test_split, KFold
import math
import random

# ====== Random Seed Initialization ====== #
def seed_everything(seed = 1024):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ============================== #
# ====== Argument Parsing ====== #
# ============================== #
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()

parser.add_argument('--model_name', type=str, default='HeteroNet_AllConcat_Top1000_231111')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--split_by', type=str, default='drug')
# parser.add_argument('--response_type', type=str, default='IC50')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1024)

parser.add_argument('--num_folds', type=int, default=5)
parser.add_argument('--k', type=int, default=1000)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--display_step', type=int, default=1500)
parser.add_argument('--testset_yes', type=bool, default=True)
parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--currentdir', type=str, default='/data/project/inyoung/DGDRP/')
parser.add_argument('--datadir', type=str, default='/data/project/inyoung/DGDRP/Data')
parser.add_argument('--root', type=str, default='/data/project/inyoung/DGDRP/Data/graph_pyg/20_indirect_targets')

parser.add_argument('--data_type', type=str, default='20_indirect_targets')
parser.add_argument('--template_adj_fname', type=str, default='template_adjacency_matrix_20_indirect_targets.tsv')

args = parser.parse_args()


response_fpath = os.path.join(args.datadir, 'response_data_total.tsv')
expression_fpath = os.path.join(args.datadir, 'expression_10k_genes_data_total.pt')
drug_fp_fpath = os.path.join(args.datadir, 'drug_data_total.pt')
template_adj_fpath = os.path.join(args.datadir, args.template_adj_fname) ####

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print('torch version: ', torch.__version__)
print(device)
seed_everything(seed=args.seed)


# =============================== #
# ====== Define Experiment ====== #
# =============================== #
from trainer import train, validate, test #, average_pcc, average_spearman
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
import json

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


# ============================ #
# ====== Run Experiment ====== #
#============================= #
name_var1 = '_'
name_var2 = '_'
list_var1 = ['_']
list_var2 = ['_']

total_results = defaultdict(list)
best_best_epoch = 0
best_best_train_loss = 99.
best_best_train_metric = 0
best_best_valid_loss = 99.
best_best_valid_metric = 0
best_var1_value = ''
best_var2_value = ''
for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        args.exp_name = f'{args.model_name}_{args.split_by}'
        args.outdir = os.path.join(args.currentdir, 'Results', args.exp_name)
        createFolder(args.outdir)
        logging(str(args), args.outdir, args.exp_name+'.log')

        # ===================== #
        # ====== Dataset ====== #
        # ===================== #
        response_data = pd.read_csv(response_fpath, sep='\t')
        response_data = response_data.reset_index().rename(columns={'index':'idx'})
        expression_data = torch.load(expression_fpath)
        drug_data = torch.load(drug_fp_fpath)
        adjacency_matrix = pd.read_csv(template_adj_fpath, sep='\t', index_col=0, header=0)

        total_dataset = DrugResponse_Dataset(response_data, expression_data, drug_data, adjacency_matrix, 
                                        root=args.root, data_type=args.data_type) # response_type=args.response_type

        # ========================================== #
        # ====== CV: Train/Valid & Test Split ====== #
        # ========================================== #
        if args.split_by == 'drug':
            split_entity_list = response_data['drug_name'].unique()
        elif args.split_by == 'cell':
            split_entity_list = response_data['cell_name'].unique()
        # elif args.split_by == 'mix' or args.split_by == 'both':
        #     split_entity_list = response_data['idx'].values
        else:
            raise ValueError('split_by should be one of [drug, cell, mix, both]')
        print("#Entity:", len(split_entity_list))

        # === Train/Valid vs. Test Split === #
        train_valid_entities, test_entities = train_test_split(split_entity_list, test_size=args.test_size, random_state=1004)

        kfold = KFold(n_splits=args.num_folds, shuffle=True)
        
        for fold, (train_entity_ids, valid_entity_ids) in enumerate(kfold.split(train_valid_entities)):

            if args.split_by == 'drug':
                test_idxs = response_data.query('drug_name in @test_entities')['idx'].to_list()

                train_drugs = train_valid_entities[train_entity_ids]
                valid_drugs = train_valid_entities[valid_entity_ids]

                train_idxs = response_data.query('drug_name in @train_drugs')['idx'].to_list()
                valid_idxs = response_data.query('drug_name in @valid_drugs')['idx'].to_list()
            elif args.split_by == 'cell':
                test_idxs = response_data.query('cell_name in @test_entities')['idx'].to_list()

                train_cells = train_valid_entities[train_entity_ids]
                valid_cells = train_valid_entities[valid_entity_ids]

                train_idxs = response_data.query('cell_name in @train_cells')['idx'].to_list()
                valid_idxs = response_data.query('cell_name in @valid_cells')['idx'].to_list()
            elif args.split_by == 'mix':
                test_idxs = list(test_entities)

                train_idxs = list(train_valid_entities[train_entity_ids])
                valid_idxs = list(train_valid_entities[valid_entity_ids])
            elif args.split_by == 'both':
                # === Cell Split === #
                cell_list = list(response_data['cell_name'].unique())
                n_train_cell = math.floor(len(cell_list) * 0.7)
                n_valid_cell = (len(cell_list) - n_train_cell) // 2
                n_test_cell = len(cell_list) - n_train_cell - n_valid_cell
                print(f"#Train Cell lines: {n_train_cell},\t #Valid Cell lines: {n_valid_cell}\t #Test Cell lines: {n_test_cell}")
                print(f"#Total: {n_train_cell + n_valid_cell + n_test_cell}")

                train_cells = random.sample(cell_list, k=n_train_cell)
                rest_cells = [cell for cell in cell_list if cell not in train_cells]
                valid_cells = random.sample(rest_cells, k=n_valid_cell)
                test_cells = [cell for cell in rest_cells if cell not in valid_cells]

                # === Drug Split === #
                drug_list = list(response_data['drug_name'].unique())
                n_train_drug = math.floor(len(drug_list) * 0.7)
                n_valid_drug = (len(drug_list) - n_train_drug) // 2
                n_test_drug = len(drug_list) - n_train_drug - n_valid_drug
                print(f"#Train Drug lines: {n_train_drug},\t #Valid Drug lines: {n_valid_drug}\t #Test Drug lines: {n_test_drug}")
                print(f"#Total: {n_train_drug + n_valid_drug + n_test_drug}")

                train_drugs = random.sample(drug_list, k=n_train_drug)
                rest_drugs = [drug for drug in drug_list if drug not in train_drugs]
                valid_drugs = random.sample(rest_drugs, k=n_valid_drug)
                test_drugs = [drug for drug in rest_drugs if drug not in valid_drugs]

                # === Get Indexes === #
                train_idxs = response_data.query('cell_name in @train_cells and drug_name in @train_drugs')['idx'].to_list()
                valid_idxs = response_data.query('cell_name in @valid_cells and drug_name in @valid_drugs')['idx'].to_list()
                test_idxs  = response_data.query('cell_name in @test_cells and drug_name in @test_drugs')['idx'].to_list()
        
            print(f'FOLD {fold}')

            # Subset the data
            train_set = torch.utils.data.Subset(total_dataset, train_idxs)
            valid_set = torch.utils.data.Subset(total_dataset, valid_idxs)
            test_set  = torch.utils.data.Subset(total_dataset, test_idxs)

            # === Data Loader === #
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
            test_loader  = DataLoader(test_set, batch_size=10, shuffle=False, drop_last=False)

            dataset_partition = {
                'train_loader': train_loader,
                'valid_loader': valid_loader,
                'test_loader' : test_loader
            }

            print("-----------TRAIN DATASET-----------")
            print("NUMBER OF DATA:", train_set.__len__())
            print("-----------VALID DATASET-----------")
            print("NUMBER OF DATA:", valid_set.__len__())
            print("-----------TEST  DATASET-----------")
            print("NUMBER OF DATA:", test_set.__len__())

            # ============= #
            # === Model === #
            # ============= #
            model = HeteroNet_EmbConcatTotal(gnn_embed_dim=128, cell_dim=10000, drug_dim=128,
                            cell_embed_dim=128, drug_embed_dim=128,
                            predictor_hid_dims=[128, 128], predictor_dropout=0.4,
                            k=args.k).to(device)
            
            # =============== #
            # === Loss fn === #
            # =============== #
            loss_fn = nn.MSELoss()

            # ====== Run Experiment ====== #
            setting, result, best_performances, model_max = experiment(name_var1, name_var2, var1, var2, args, dataset_partition, model, loss_fn, device)
            save_exp_result(setting, result, args.outdir)
            
            if best_performances['best_valid_corr'] >= best_best_valid_metric:
                best_best_epoch = best_performances['best_epoch']
                best_best_train_loss = best_performances['best_train_loss']
                best_best_train_metric = best_performances['best_train_corr']
                best_best_valid_loss = best_performances['best_valid_loss']
                best_best_valid_metric = best_performances['best_valid_corr']
                best_var1_value = var1
                best_var2_value = var2
                best_setting = setting
                best_result = result
            
            total_results[name_var1].append(var1)
            total_results[name_var2].append(var2)
            total_results['best_epoch'].append(best_performances['best_epoch'])
            total_results['best_train_loss'].append(best_performances['best_train_loss'])
            total_results['best_train_corr'].append(best_performances['best_train_corr'])
            total_results['best_valid_loss'].append(best_performances['best_valid_loss'])
            total_results['best_valid_corr'].append(best_performances['best_valid_corr'])
        
print(f'Best Train Loss: {best_best_train_loss:.4f}')
print(f'Best Train Corr: {best_best_train_metric:.4f}')
print(f'Best Valid Loss: {best_best_valid_loss:.4f}')
print(f'Best Valid Corr: {best_best_valid_metric:.4f}')

