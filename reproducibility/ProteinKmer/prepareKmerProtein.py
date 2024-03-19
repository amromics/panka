## Run ML methods on PanPred and panta outputs 
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import datasets
from sklearn import svm
import random
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from Bio import SeqIO
from skbio import Sequence
import timeit
from pangraph.utils import binary_label
from sklearn.feature_selection import mutual_info_classif, chi2

# E. coli
# pa_matrix = pd.read_csv('output/pantaEcoli1936/gene_presence_absence.Rtab', sep='\t', index_col=0).T
# pa_matrix = pd.read_csv('output/pantaEcoli1936align'+version+'/gene_presence_absence.Rtab', sep='\t', index_col=0).T
# metadata_panta = pd.read_csv("data/Ecoli1936metafiles/metadata_final.csv")

# K. p
pa_matrix = pd.read_csv('output/pantaKpPatric/gene_presence_absence.Rtab', sep='\t', index_col=0).T
metadata_panta = pd.read_csv("data/Kpmetadata_final.csv")
interval_len = round(pa_matrix.shape[0]/3)
print("interval_len = ", interval_len)
sample_list = list(pa_matrix.index)
sample_list_faa = [sample + '.faa' for sample in sample_list] #, 'SAMEA2204229.contig.fna'

ksize = 10
data_name = 'KpPatric' # 'Ecoli1936'

## Split data into folds to avoid lack of memory.

for aafold in ['0', '1', '2']:
    print('aafold = ', aafold)
    # aafold = '0' #, '1', '2' (make a loop for aafold)
    ## Divide into 3 folds
    if aafold == '0':
        start_idx = 0; end_idx = interval_len
    elif aafold == '1':
        start_idx = interval_len; end_idx = 2*interval_len
    else:
        start_idx = 2*interval_len; end_idx = metadata_panta.shape[0]
    
    kmer_seq_set = set()
    sample_idx = 0
    pairdata = []
    # for seq_idx in sample_list_faa:
    for seq_idx in range(start_idx, end_idx):
        # print(seq_id)
        # start = timeit.default_timer()
        # data_dir = 'data/Ecoli1936/prokkaMore/'+sample_list_faa[seq_idx]
        # Directory of K. p. prokka results
        data_dir = 'data/KpPatric/prokkaMore/'+sample_list_faa[seq_idx]
        print(data_dir)
        kmer_seq = []
        fasta_sequences = SeqIO.parse(open(data_dir),'fasta')
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            n_kmers = len(sequence) - ksize + 1
            for i in range(n_kmers):
                kmer_seq.append(sequence[i:i + ksize])
                pairdata.append((sample_idx, sequence[i:i + ksize]))

        sample_idx += 1
        kmer_seq_set.update(kmer_seq)

    kmer_seq_set = list(kmer_seq_set)
    kmer2index = {}
    for i in range(len(kmer_seq_set)):
        kmer2index[kmer_seq_set[i]] = i

    # kmer_matrix = np.zeros((n_samples, len(kmer_seq_set)))
    kmer_matrix = np.zeros((sample_idx, len(kmer_seq_set)), dtype = np.int)
    for idx, kmer in pairdata:
        kmer_matrix[idx, kmer2index[kmer]] = 1

    selector = VarianceThreshold(threshold=0.01)
    kmer_matrix_VT = selector.fit_transform(kmer_matrix)
    selected_features = np.array([kmer_seq_set[idx] for idx in selector.get_support(indices=True)])

    # Save the data
    # np.save('data/kmer_Fold'+data_fold+'_mat_VT1.npy', kmer_matrix_VT) # save numpy array
    # np.save('data/kmer_Fold'+data_fold+'_mat_VT1_features.npy', selected_features) # save numpy array

    # ## Feature selection Phase II: select features that correlate with the labels
    mutual_mat = []
    for idx in range(2, metadata_panta.shape[1]):
        # y_class = metadata_panta.iloc[:600,idx].values
        y_class = metadata_panta.iloc[start_idx:end_idx, idx].values
        print(metadata_panta.columns[idx])
        y, nonenan_index = binary_label(y_class) # v6
        pa_matrix_new = kmer_matrix_VT[nonenan_index, ]
        y_new = y[nonenan_index].astype(int)
        if len(y_new) > 10:
            scores, pvalue = chi2(pa_matrix_new, y_new)
            mutual_mat.append(scores)
    mutual_mat = np.array(mutual_mat)
    mutual_mat_mean = mutual_mat.mean(axis=0)

    top_features = np.argsort(mutual_mat_mean)[::-1][:100000]
    kmer_matrix_VT_top_features = kmer_matrix_VT[:,top_features]
    selected_features_top = selected_features[top_features]
    np.save('data/kmer_k'+str(ksize)+'_'+data_name+'_Fold'+aafold+'_mat_VT1_top_features.npy', kmer_matrix_VT_top_features) # save numpy array
    np.save('data/kmer_k'+str(ksize)+'_'+data_name+'_Fold'+aafold+'_mat_VT1_features_top_features.npy', selected_features_top) # save numpy array
## end of the loop

# # Concat 3 datasets together (aafold for 0, 1, 2 and ...)
data0 = np.load('data/kmer_k'+str(ksize)+'_'+data_name+'_Fold0_mat_VT1_top_features.npy') # save numpy array
feature0 = np.load('data/kmer_k'+str(ksize)+'_'+data_name+'_Fold0_mat_VT1_features_top_features.npy') # save numpy array
df0 = pd.DataFrame(data = data0, columns = feature0)
#
data1 = np.load('data/kmer_k'+str(ksize)+'_'+data_name+'_Fold1_mat_VT1_top_features.npy') # save numpy array
feature1 = np.load('data/kmer_k'+str(ksize)+'_'+data_name+'_Fold1_mat_VT1_features_top_features.npy') # save numpy array
df1 = pd.DataFrame(data = data1, columns = feature1)
#
data2 = np.load('data/kmer_k'+str(ksize)+'_'+data_name+'_Fold2_mat_VT1_top_features.npy') # save numpy array
feature2 = np.load('data/kmer_k'+str(ksize)+'_'+data_name+'_Fold2_mat_VT1_features_top_features.npy') # save numpy array
df2 = pd.DataFrame(data = data2, columns = feature2)

df_full = pd.concat([df0,df1,df2], axis=0, ignore_index=True) 
df_full = df_full.fillna(0)
snp_mat = df_full.values

np.save('data/kmer_k'+str(ksize)+'_'+data_name+'_full_mat_VT1_AA.npy', snp_mat) # save numpy array

