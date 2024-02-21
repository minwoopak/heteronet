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
from collections import defaultdict

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

parser.add_argument('--model_name', type=str, default='MLP_High_Variance_v2')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--split_by', type=str, default='drug')
parser.add_argument('--response_type', type=str, default='IC50')
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--seed', type=int, default=1024)

parser.add_argument('--drug_dim', type=int, default=128)
parser.add_argument('--cell_dim', type=int, default=10000)
parser.add_argument('--num_folds', type=int, default=5)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--display_step', type=int, default=1500)
parser.add_argument('--testset_yes', type=bool, default=True)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--currentdir', type=str, default='/data/project/minwoo/feature_selection/phase_5_selection_method_comparison_split_corrected')
parser.add_argument('--datadir', type=str, default='/data/project/minwoo/Data')

parser.add_argument('--drug_embed_dim', type=int, default=64)
parser.add_argument('--cell_embed_dim', type=int, default=64)
parser.add_argument('--embed_hid_dim', type=list, default=[128, 128])
parser.add_argument('--fc_hid_dim', type=list, default=[128, 128])

args = parser.parse_args()


response_fpath = os.path.join(args.datadir, 'drug_response', 'response_data_total.tsv')
expression_fpath = os.path.join(args.datadir, 'drug_response', 'expression_highVariance_genes_data_total.pt')
drug_fp_fpath = os.path.join(args.datadir, 'drug_response', 'drug_data_total.pt')

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print('torch version: ', torch.__version__)
print(device)
seed_everything(seed=args.seed)


# =================== #
# ====== Model ====== #
# =================== #
from model import DRP_MLP


# ===================== #
# ====== Dataset ====== #
# ===================== #
from dataset import DRPDataset


# ============================ #
# ====== Run Experiment ====== #
#============================= #
from trainer import experiment
from utils import logging
from torch.utils.data import DataLoader
from utils import save_exp_result
from sklearn.model_selection import train_test_split, KFold
import math
import random


name_var1 = 'response_type'
name_var2 = '_'
list_var1 = ['IC50']
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

        total_dataset = DRPDataset(response_data=response_data, 
                                   expression_data=expression_data, 
                                   drug_data=drug_data, 
                                   response_type=args.response_type)

        # ========================================== #
        # ====== CV: Train/Valid & Test Split ====== #
        # ========================================== #
        if args.split_by == 'cell':
            cell_list = list(response_data['cell_name'].unique())

            n_train_cell = math.floor(len(cell_list) * (1-args.test_size))
            n_test_cell  = len(cell_list) - n_train_cell

            print(f"#Train Cell lines: {n_train_cell}\t #Test Cell lines: {n_test_cell}")
            print(f"#Total: {n_train_cell + n_test_cell}")

            train_valid_entities = random.sample(cell_list, k=n_train_cell)
            test_entities  = [cell for cell in cell_list if cell not in train_valid_entities]

            assert len(set(train_valid_entities) & set(test_entities)) == 0

        elif args.split_by == 'drug':
            drug_list = list(response_data['drug_name'].unique())
            n_train_drug = math.floor(len(drug_list) * (1-args.test_size))
            n_test_drug  = len(drug_list) - n_train_drug

            print(f"#Train Drug lines: {n_train_drug},\t #Test Drug lines: {n_test_drug}")
            print(f"#Total: {n_train_drug + n_test_drug}")

            train_valid_entities = random.sample(drug_list, k=n_train_drug)
            test_entities = [drug for drug in drug_list if drug not in train_valid_entities]

            assert len(set(train_valid_entities) & set(test_entities)) == 0
        else:
            raise ValueError('split_by should be one of [drug, cell]')
        print(f"Splitting by {args.split_by}...")


        kfold = KFold(n_splits=args.num_folds, shuffle=True)
                
        for fold, (train_entity_ids, valid_entity_ids) in enumerate(kfold.split(train_valid_entities)):
            if args.split_by == 'drug':
                test_idxs = response_data.query('drug_name in @test_entities')['idx'].to_list()

                train_drugs = np.array(train_valid_entities)[train_entity_ids]
                valid_drugs = np.array(train_valid_entities)[valid_entity_ids]

                train_idxs = response_data.query('drug_name in @train_drugs')['idx'].to_list()
                valid_idxs = response_data.query('drug_name in @valid_drugs')['idx'].to_list()

            elif args.split_by == 'cell':
                test_idxs = response_data.query('cell_name in @test_entities')['idx'].to_list()

                train_cells = np.array(train_valid_entities)[train_entity_ids]
                valid_cells = np.array(train_valid_entities)[valid_entity_ids]

                train_idxs = response_data.query('cell_name in @train_cells')['idx'].to_list()
                valid_idxs = response_data.query('cell_name in @valid_cells')['idx'].to_list()
        
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
            model = DRP_MLP(drug_dim=args.drug_dim,
                            cell_dim=args.cell_dim,
                            drug_embed_dim=args.drug_embed_dim,
                            cell_embed_dim=args.cell_embed_dim,
                            embed_hid_dim=args.embed_hid_dim,
                            fc_hid_dim=args.fc_hid_dim).to(device)
            
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
