import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
print(os.popen("pwd").read())
import argparse
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from collections import defaultdict
import gseapy as gp

parser.add_argument('--workdir', type=str, default='/data/project/inyoung/DGDRP')
parser.add_argument('--n_indirect_targets', type=int, default=20)

args = parser.parse_args()

args.datadir = os.path.join(args.workdir, 'Data')


# ================================================ #
# ====== Heterogeneous network construction ====== #
# ================================================ #

# === Import PS profile === #
# drug - genes : netgp scores
args.ps_profile_fpath = os.path.join(args.datadir, 'drug_target_profile_original.tsv')
ps_profile = pd.read_csv(args.ps_profile_fpath, sep='\t')
drug_list = ps_profile['drug_name'].to_list()
ps_profile = ps_profile.set_index('drug_name')

# === Import PPI Template === #
args.ppi_template_fpath = os.path.join(args.datadir, '9606.protein.links.symbols.v11.5.txt')
ppi = pd.read_csv(args.ppi_template_fpath, sep='\t')
ppi_genes = list(set(ppi['source'].to_list() + ppi['target'].to_list()))
print('#PPI Genes:', len(ppi_genes))

# === Import Drug Target Info === #
# drugbank + GDSC target information
args.drug_target_fpath = os.path.join(args.datadir, 'dti_info_final_common_drugs_only.tsv')
drug_target_info = pd.read_csv(args.drug_target_fpath, sep='\t')
drug_target_info = drug_target_info[['drug_name', 'gene_name']]
drug_target_info = drug_target_info[drug_target_info['drug_name'].isin(ps_profile.index)]
drug_target_info = drug_target_info[drug_target_info['gene_name'].isin(ppi_genes)]

args.resultdir = os.path.join(args.datadir, f'drug_networks_{args.n_indirect_targets}_indirect_targets')
createFolder(args.resultdir)


# ================================================ #
# ====== 1) Direct Target - Indirect Target ====== #
# ================================================ #
# === For each drug: get Top 100 ps genes (indirect targets) : Save edges === #
lvl1_edges = defaultdict(list)
for drug in drug_list:
    # === Direct Targets === #
    direct_targets = drug_target_info.query('drug_name == @drug')['gene_name'].to_list()

    # === Indirect Targets === #
    drug_ps_profile = ps_profile.loc[drug]
    indirect_targets = drug_ps_profile.sort_values(ascending=False).head(args.n_indirect_targets).index.to_list()

    for direct_target in direct_targets:
        for indirect_target in indirect_targets:
            lvl1_edges['direct'].append(direct_target)
            lvl1_edges['indirect'].append(indirect_target)

lvl1_edges_df = pd.DataFrame(lvl1_edges).drop_duplicates()
print(f"All direct/indirect gene pairs: {lvl1_edges_df.shape[0]:,}")

# === leave only the edges that are in the PPI network === #
ppi_edges = pd.concat([
    pd.DataFrame(ppi[['source', 'target']].values), 
    pd.DataFrame(ppi[['target', 'source']].values)
]).drop_duplicates()
ppi_edges.columns = ['direct', 'indirect']

lvl1_edges_df = pd.merge(lvl1_edges_df, ppi_edges, on=['direct', 'indirect'], how='inner')
print(f"PPI-included direct/indirect gene pairs: {lvl1_edges_df.shape[0]:,}")

lvl1_edges_df = pd.concat([
    pd.DataFrame(lvl1_edges_df[['direct', 'indirect']].values),
    pd.DataFrame(lvl1_edges_df[['indirect', 'direct']].values)
]).drop_duplicates()
print(f"Undirected PPI-included direct/indirect gene pairs: {lvl1_edges_df.shape[0]:,}")
lvl1_edges_df.columns = ['source', 'target']


# ================================================= #
# ====== 2) Direct/Indirect Target - Pathway ====== #
# ================================================= #
# gp.get_library_name()
# ['GO_Biological_Process_2023', 'GO_Cellular_Component_2023' 'GO_Molecular_Function_2023', 'KEGG_2021_Human', 'Reactome_2022']
kegg_gmt = gp.parser.get_library('KEGG_2021_Human', 
								  organism='Human', 
								  min_size=3, 
								  max_size=2000, 
								  gene_list=None)
print('KEGG #Terms:\t', len(kegg_gmt.keys()))
kegg_genes = [gene for genes in kegg_gmt.values() for gene in genes]
print("KEGG #Genes:\t", len(set(kegg_genes)))

# === Filter KEGG genes by Lvl1 edges genes === #
total_genes = list(pd.concat([lvl1_edges_df['source'], lvl1_edges_df['target']]).unique())

kegg_df = defaultdict(list)
for term, genes in kegg_gmt.items():
    for gene in genes:
        if gene not in total_genes:
            continue
        kegg_df['term'].append(term)
        kegg_df['gene'].append(gene)
kegg_df = pd.DataFrame(kegg_df)
kegg_df.columns = ['source', 'target']

# ============================ #
# ====== 3) Merge Graph ====== #
# ============================ #
trial1_nwk = pd.concat([lvl1_edges_df, kegg_df], axis=0)
print(lvl1_edges_df.shape, kegg_df.shape)

# ======================================== #
# ====== Construct Adjacency Matrix ====== #
# ======================================== #
nodes = pd.Series(pd.concat([trial1_nwk['source'], trial1_nwk['target']]).unique())

# create empty adjacency matrix filled with zeros with the number of nodes as dimension
adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
# add self-loops
for node in nodes:
    adjacency_matrix.loc[node, node] = 1


adj_fpath = os.path.join(args.datadir, f'template_adjacency_matrix_{args.n_indirect_targets}_indirect_targets.tsv')
adjacency_matrix.to_csv(adj_fpath, sep='\t', index=True, header=True)


# ========================================== #
# ====== Create Drug-Specific Network ====== #
# ========================================== #
for drug in drug_list:
    # =========================== #
    # ====== For Each Drug ====== #
    # =========================== #
    drug_lvl1_edges = defaultdict(list)

    # === Create empty adjacency matrix filled with zeros with the number of nodes as dimension === #
    adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    # add self-loops
    for node in nodes:
        adjacency_matrix.loc[node, node] = 1

    # ====== Level 1 Edges ====== #
    # === 1. Define Direct Targets === #
    direct_targets = drug_target_info.query('drug_name == @drug')['gene_name'].to_list()

    # === 2. Define Indirect Targets === #
    drug_ps_profile = ps_profile.loc[drug]
    indirect_targets = drug_ps_profile.sort_values(ascending=False).iloc[:args.n_indirect_targets].index.to_list()

    for direct_target in direct_targets:
        for indirect_target in indirect_targets:
            drug_lvl1_edges['direct'].append(direct_target)
            drug_lvl1_edges['indirect'].append(indirect_target)

    drug_lvl1_edges_df = pd.DataFrame(drug_lvl1_edges).drop_duplicates()
    print(f"All direct/indirect gene pairs: {drug_lvl1_edges_df.shape[0]:,}")

    # === 3. Define connections: Leave only the edges that are in the PPI network === #
    ppi_edges = pd.concat([
        pd.DataFrame(ppi[['source', 'target']].values), 
        pd.DataFrame(ppi[['target', 'source']].values)
    ]).drop_duplicates()

    ppi_edges.columns = ['direct', 'indirect']
    drug_lvl1_edges_df = pd.merge(drug_lvl1_edges_df, ppi_edges, on=['direct', 'indirect'], how='inner')
    print(f"PPI-included direct/indirect gene pairs: {drug_lvl1_edges_df.shape[0]:,}")

    drug_lvl1_edges_df = pd.concat([
        pd.DataFrame(drug_lvl1_edges_df[['direct', 'indirect']].values),
        pd.DataFrame(drug_lvl1_edges_df[['indirect', 'direct']].values)
    ]).drop_duplicates()
    print(f"Undirected PPI-included direct/indirect gene pairs: {drug_lvl1_edges_df.shape[0]:,}")
    drug_lvl1_edges_df.columns = ['source', 'target']

    # ====== Level 2 Edges ====== #
    # === 1. Pathways of the Indirect Targets === #
    drug_lvl2_edges_df = kegg_df    # .query('target in @indirect_targets')
    drug_lvl2_edges_df = pd.concat([
        pd.DataFrame(drug_lvl2_edges_df[['source', 'target']].values),
        pd.DataFrame(drug_lvl2_edges_df[['target', 'source']].values)
    ]).drop_duplicates()
    drug_lvl2_edges_df.columns = ['source', 'target']

    # === 2. Save Edges: Indirect Targets - Pathways === #
    drug_specific_edges = pd.concat([drug_lvl1_edges_df, drug_lvl2_edges_df], axis=0)

    # ====== Fill in the adjacency matrix with the edges from the drug-specific edges (undirected) ====== #
    for node1, node2 in drug_specific_edges.values:
        adjacency_matrix.loc[node1, node2] = 1

    drug_nwk_fpath = os.path.join(args.resultdir, f'{drug}_adjacency_matrix_{args.n_indirect_targets}_indirect_targets.tsv')
    adjacency_matrix.to_csv(drug_nwk_fpath, sep='\t', index=True, header=True)

