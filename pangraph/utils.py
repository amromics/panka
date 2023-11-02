import re
from os.path import join, exists 
import numpy as np
import networkx as nx
from networkx import NetworkXNoPath
import json
import os
import pandas as pd
import gzip

def help_fnc(i, j):
    nonmatch = True
    for ele in range(min(127,len(j)), 63, -1):
        if i.endswith(j[:ele]):
            # print(ele, end =":")
            nonmatch = False
            return j[ele:]
    if nonmatch:
        return j
        
def overlap(a, b):
    test_list = [a, b]
    res = ''.join(help_fnc(i, j) for i, j in zip([''] + test_list, test_list))
    return(res)

def max_common_subsequence(a, b):
    result = np.zeros((len(a), len(b)))
    n = max(len(a), len(b))
    max_value = 0; best_max = 0
    for i in range(len(a)):
        for j in range(len(b)):
            if (a[i]==b[j]):
                result[i,j] = result[i-1,j-1]+1 #remember to initialize the borders with zeros
            else:
                result[i,j]=0
    return(np.amax(result)/n)

from difflib import SequenceMatcher
def similarity_sequence(a, b):
    return (SequenceMatcher(None, a, b).ratio())

def _convert_edgeStr (_edgeStr):
    assert isinstance(_edgeStr, str) and 'EDGE_' in _edgeStr, "{} is not legal component from spades FASTG".format(edgeString)
    _edgeId = _edgeStr.split('_')[1]
    _edgeDirection = '-' if "'" in _edgeStr else '+'
    return f'{_edgeId}{_edgeDirection}'

def _get_node_length (_edgeStr):
    assert isinstance(_edgeStr, str) and 'EDGE_' in _edgeStr, "{} is not legal component from spades FASTG".format(edgeString)
    _edgeId = _edgeStr.split('_')[3]
    return int(_edgeId)
def append_strand(nodeStr):
    if "'" in nodeStr:
        return nodeStr[:-1] + '-'
    else:
        return nodeStr + '+'
def append_strand_reverse(nodeStr):
    if "'" in nodeStr:
        return nodeStr[:-1] + '+'
    else:
        return nodeStr + '-'

def getContigsAdjacency(spadesOutDir=None, graphFilePath=None, pathFilePath=None):
    if graphFilePath == None and spadesOutDir != None:
        graphFilePath = join(spadesOutDir, 'assembly_graph.fastg')
    if pathFilePath == None and spadesOutDir != None:
        pathFilePath = join(spadesOutDir, 'contigs.paths')    
        
    assert exists(graphFilePath), "FASTG file not found"    
    assert exists(pathFilePath), "Contig paths file not found"    

    #Init set of link and dicts of endpoint
    componentLinks = set()
    contigLinks = set()
    startDict = {}
    endDict = {}
    #Read links between components from assembly graph FASTG file
    graph_file = open(graphFilePath, 'r')
    for _line in graph_file.readlines():
        if(_line.startswith('>')):
            _edges_list = re.split(':|,', _line.strip());
            if(len(_edges_list) <2):
                continue
            else:
                #print("From {} to {}".format(_edges_list[0], _edges_list[1:]))
                root = _convert_edgeStr(_edges_list[0])
                for _edge in _edges_list[1:]:
                    componentLinks.add((root, _convert_edgeStr(_edge)))
                    #print("{},{}".format(root, _convert_edgeStr(_edge)))                
    graph_file.close()
    #Read endpoints of each path (CONTIG) consisting of above components
    path_file = open(pathFilePath, 'r')
    for _line in path_file.readlines():
        if(_line.startswith('NODE_')):
            ctg = _line.strip()
        else:
            _edges_from_path = _line.strip().split(',')
            _start = _edges_from_path[0]; _end = _edges_from_path[-1]
            if _start in startDict.keys():
                startDict[_start].add(ctg)
            else:
                startDict[_start] = {ctg}

            if _end in endDict.keys():
                endDict[_end].add(ctg)
            else:
                endDict[_end] = {ctg}

    path_file.close()

    #Output all possible links between paths (contigs)
    # <------leftCtg---------/-start-/-> <-/-end-/---rightCtg----->
    #       
    for start,end in componentLinks:
        #print("{} to {}".format(start,end))
        if (start in endDict.keys()) and (end in startDict.keys()):
            for leftCtg in endDict[start]:
                for rightCtg in startDict[end]:
                    contigLinks.add((leftCtg, rightCtg))
    return contigLinks

def getContigsAdjacency_v2(spadesOutDir=None, graphFilePath=None, pathFilePath=None):
    if graphFilePath == None and spadesOutDir != None:
        graphFilePath = join(spadesOutDir, 'assembly_graph.fastg')
    if pathFilePath == None and spadesOutDir != None:
        pathFilePath = join(spadesOutDir, 'contigs.paths')    
        
    assert exists(graphFilePath), "FASTG file not found"    
    assert exists(pathFilePath), "Contig paths file not found"    

    #Init set of link and dicts of endpoint
    componentLinks = set()
    contigLinks = set()
    startDict = {}
    endDict = {}
    nodeweightDict = {}
    #Read links between components from assembly graph FASTG file
    graph_file = open(graphFilePath, 'r')
    for _line in graph_file.readlines():
        if(_line.startswith('>')):
            _edges_list = re.split(':|,', _line.strip());
            if(len(_edges_list) <2):
                continue
            else:
                #print("From {} to {}".format(_edges_list[0], _edges_list[1:]))
                root = _convert_edgeStr(_edges_list[0])
                for _edge in _edges_list[1:]:
                    componentLinks.add((root, _convert_edgeStr(_edge)))
                    nodeweightDict[_convert_edgeStr(_edge)] = int(_get_node_length(_edge))
                    #print("{},{}".format(root, _convert_edgeStr(_edge)))                
    graph_file.close()
    #Read endpoints of each path (CONTIG) consisting of above components
    path_file = open(pathFilePath, 'r')
    contigs_list = []
    for _line in path_file.readlines():
        if(_line.startswith('NODE_')):
            ctg = _line.strip()
            contigs_list.append(ctg)
        else:
            _edges_from_path = _line.strip().split(',')
            _start = _edges_from_path[0]; _end = _edges_from_path[-1]
            startDict[ctg] = _start
            endDict[ctg] = _end

    path_file.close()
    # for start,end in componentLinks:
    #     if (start in endDict.keys()) and (end in startDict.keys()):
    #         for leftCtg in endDict[start]:
    #             for rightCtg in startDict[end]:
    #                 contigLinks.add((leftCtg, rightCtg))
    # return contigLinks
    assembly_graph= nx.DiGraph()
    assembly_graph.add_edges_from(list(componentLinks))
    n_contigs = len(contigs_list)
    # print(nodeweightDict)
    edges_list = []
    weight_vec = []
    for i in range(n_contigs):
        for j in range(i+1, n_contigs):
            leftCtg = contigs_list[i]
            rightCtg = contigs_list[j]
            # print(leftCtg, rightCtg)
            if leftCtg[:12] != rightCtg[:12]:
                endLeftCtg = endDict[leftCtg]
                startrightCtg = startDict[rightCtg]
                if assembly_graph.has_node(endLeftCtg) and assembly_graph.has_node(startrightCtg):
                    path_nucleotides = 0
                    path_len = -1
                    try:
                        path = nx.shortest_path(assembly_graph, source=endLeftCtg, target=startrightCtg)
                        for idxp in range(1, len(path)-1):
                            path_nucleotides = path_nucleotides + nodeweightDict[path[idxp]]
                        path_len = len(path)
                    except NetworkXNoPath:
                        path = None
                    if path != None and path_nucleotides <= 500:
                    # if path != None and path_len <= 2: # =2 <=> directed graph
                        contigLinks.add((leftCtg, rightCtg))
                    if path != None:
                        edges_list.append((append_strand(leftCtg), append_strand(rightCtg)))
                        weight_vec.append(path_nucleotides)

    weight_vec = np.array(weight_vec)
    weight_vec = np.interp(weight_vec, (weight_vec.min(), weight_vec.max()), (1.0, 10.0))
    wp_vec = [(edges_list[i][0], edges_list[i][1], weight_vec[i]) for i in range(len(weight_vec))]
    weighted_CG = nx.DiGraph()
    weighted_CG.add_weighted_edges_from(wp_vec)
    return (contigLinks, weighted_CG)


def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', '_':'_','*':'*'}
    # seq = "TCGGGCCC"
    reverse_complement = "".join(complement.get(base, base) for base in reversed(seq))
    return(reverse_complement)

def getOverlapLength(i, j):
    for ele in range(min(500,len(j)), -1, -1):
        if i.endswith(j[:ele]):
            return ele
    return 0

def append_strand(input_str):
    if input_str[-1]=="'":
        return (input_str[:-1] + '-')
    else:
        return (input_str + '+')

def append_strand_undirected(input_str):
    # print("Undirected!!!")
    if input_str[-1]=="'":
        return (input_str[:-1])
    else:
        return (input_str)
    
def buildOverlapEdge(contigs, min_overlap=30, graph = "directed"):
    # contigs: dict of contigs (key = contig id, value = sequence)
    # min_overlap: minimum
    # return: a list of overlap edges
    ## reimplementing the reverse_complement: not all need to be computed.
    edge_list_overlap = []
    if graph=='directed':
        for key1 in contigs:
            for key2 in contigs:
                if key1 != key2:
                    if getOverlapLength(contigs[key1], contigs[key2]) >= min_overlap:
                        edge_list_overlap.append((key1 + '+', key2 + '+'))
                        edge_list_overlap.append((key2 + '-', key1 + '-'))
                    minsize = min(len(contigs[key2]), 500)
                    if getOverlapLength(contigs[key1], reverse_complement(contigs[key2][-minsize:-1])) >= min_overlap:
                        edge_list_overlap.append((key1+ '+', key2 + '-'))
                        edge_list_overlap.append((key2 + '+', key1 + '-'))
    else:
        for key1 in contigs:
            for key2 in contigs:
                if key1 != key2:
                    if getOverlapLength(contigs[key1], contigs[key2]) >= min_overlap:
                        edge_list_overlap.append((key1, key2))
                    minsize = min(len(contigs[key2]), 500)
                    if getOverlapLength(contigs[key1], reverse_complement(contigs[key2][-minsize:-1])) >= min_overlap:
                        edge_list_overlap.append((key1, key2))
                    
    return(edge_list_overlap)             

def read_contigs2dict(data_dir):
    # Read the contigs
    # https://coding4medicine.com/backup/Python/reading-fasta-files.html
    # f=open('/data/hoan/amromics/simulation/art_output/spades_output/contigs.fasta','r')
    f=open(data_dir,'r')
    lines=f.readlines()

    hre=re.compile('>(\S+)')
    lre=re.compile('^(\S+)$')

    gene={}

    for line in lines:
            outh = hre.search(line)
            if outh:
                    id=outh.group(1)
            else:
                    outl=lre.search(line)
                    if(id in gene.keys()):
                            gene[id] += outl.group(1)
                    else:
                            gene[id]  =outl.group(1)
                            
    return gene

def write_fasta(dictionary, filename):
    """
    Takes a dictionary and writes it to a fasta file
    Must specify the filename when caling the function
    https://www.programcreek.com/python/?CodeExample=write+fasta
    """

    import textwrap
    with open(filename, "w") as outfile:
        for key, value in dictionary.items():
            outfile.write(">")
            outfile.write(key + "\n")
            outfile.write("\n".join(textwrap.wrap(value, 60)))
            outfile.write("\n")

    print("Success! File written")
    
def generate_fasta_from_dict(gene, adj_list_assembly, option='partial'):
    # option (all or partial), all if generate all contigs, partial only joined contigs.
    gene_origin = {}
    if option =='partial':   
        for key in adj_list_assembly:
            path = adj_list_assembly[key]
            if len(path) > 1:
                # source_node_key = key[:-1]
                # if key[-1]=='-':
                #     gene_origin[source_node_key] = reverse_complement(gene[source_node_key])
                # else:
                #     gene_origin[source_node_key] = gene[source_node_key]
                # for i in range(1, len(path)):
                source_node_key = key
                gene_origin[source_node_key] = ''
                for i in range(len(path)):
                    next_neighbor_temp = path[i][:-1]
                    strand = path[i][-1]
                    if next_neighbor_temp in gene:
                        if strand == '+':
                            # print(source_node_key)
                            # gene_origin[source_node_key] = gene_origin[source_node_key]+gene[next_neighbor_temp]
                            gene_origin[source_node_key] = overlap(gene_origin[source_node_key], gene[next_neighbor_temp])
                        else:
                            # gene_origin[source_node_key] = gene_origin[source_node_key]+reverse_complement(gene[next_neighbor_temp])
                            gene_origin[source_node_key] = overlap(gene_origin[source_node_key], reverse_complement(gene[next_neighbor_temp]))
    elif option=='all':
        # ## Keep all contigs
        gene_origin = gene.copy()
        # adj_list = {}
        for key in adj_list_assembly:
            path = adj_list_assembly[key]
            if len(path) > 1:
                source_node_key = key[:-1]
                if key[-1]=='-':
                    gene_origin[source_node_key] = reverse_complement(gene[source_node_key])

                if source_node_key not in gene_origin:
                    gene_origin[source_node_key] = gene[source_node_key]

                for i in range(1, len(path)):
                    # print(next_neighbor_temp)
                    next_neighbor_temp = path[i][:-1]
                    strand = path[i][-1]
                    if strand == '+':
                        # print(source_node_key)
                        # gene_origin[source_node_key] = gene_origin[source_node_key]+gene[next_neighbor_temp]
                        gene_origin[source_node_key] = overlap(gene_origin[source_node_key], gene[next_neighbor_temp])
                    else:
                        # gene_origin[source_node_key] = gene_origin[source_node_key]+reverse_complement(gene[next_neighbor_temp])
                        gene_origin[source_node_key] = overlap(gene_origin[source_node_key], reverse_complement(gene[next_neighbor_temp]))
                    if next_neighbor_temp in gene_origin:
                        del gene_origin[next_neighbor_temp]
                        
    else:
        print("Don't support!!!")
                
    return(gene_origin)


def export_metadata(path_out_pangenome):
    # print("Hello VN")
    dict_samples=json.load(open(os.path.join(path_out_pangenome,'samples.json')))
    #print(dict_samples)
    map_stringid_to_numberic_id={}
    for i in range(len(dict_samples)):
        #print(dict_samples[i]['id'])
        map_stringid_to_numberic_id[dict_samples[i]['id']]=i
    # print(map_stringid_to_numberic_id)
    dict_clusters=json.load(open(os.path.join(path_out_pangenome,'clusters.json')))
    map_geneid_to_numberic_cid={}
    cindex=0
    for k in dict_clusters.keys():
        # print(k)
        # print(dict_clusters[k])
        map_geneid_to_numberic_cid[k] =cindex

        for g in dict_clusters[k]:
            map_geneid_to_numberic_cid[g]=cindex
        cindex=cindex+1
        #map_stringid_to_numberic_id[dict_samples[i]['id']]=i
    # print(map_geneid_to_numberic_cid)
    #map gene to strain
    df_annotations= pd.read_csv(os.path.join(path_out_pangenome,'gene_annotation.csv'))
    # print(df_annotations.head())
    map_geneid_to_info={}
    for index, row  in df_annotations.iterrows():
        strand='1'
        if row['strand']=='-':
            strand='-1'
        map_geneid_to_info[row['gene_id']]={'sample_id':row['sample_id'],'seq_id':row['seq_id'],'length':row['length'],'strand':strand}
    # print(map_geneid_to_info)
    #samples.tsv
    with open(os.path.join(path_out_pangenome,'samples.tsv'), 'w') as sampletsv:
        #sampletsv.write('Name\tSampleID\n')
        for k in map_stringid_to_numberic_id.keys():
            sampletsv.write(f'{k}\t{map_stringid_to_numberic_id[k]}\n')
    #gene_info.tsv
    with open(os.path.join(path_out_pangenome,'gene_info.tsv'), 'w') as genetsv:
        #genetsv.write('GeneName\tSampleID\tclusterID\n')
        for k in map_geneid_to_info.keys():
            # print(k, map_geneid_to_info[k]['strand'])
            # print(str(map_stringid_to_numberic_id[map_geneid_to_info[k]['sample_id']]))
            # print("l", map_geneid_to_numberic_cid[k])
            genetsv.write(k+'@'+map_geneid_to_info[k]['strand']+'\t'+str(map_stringid_to_numberic_id[map_geneid_to_info[k]['sample_id']])+'\t'+str(map_geneid_to_numberic_cid[k])+'\n')
            
    # with open(os.path.join(path_out_pangenome,'gene_position.tsv'), 'w') as genepos_tsv,  gzip.open(os.path.join(path_out_pangenome,'gene_position.csv'), 'rt') as genepos_csv:
    with open(os.path.join(path_out_pangenome,'gene_position.tsv'), 'w') as genepos_tsv,  open(os.path.join(path_out_pangenome,'gene_position.csv'), 'rt') as genepos_csv:
        lines = genepos_csv.readlines()
        #genepos_tsv.write('SampleID\tContigName\tGeneSequence\n')
        for line in lines:
            row=line[:-1].split(',')

            genepos_tsv.write(f'{map_stringid_to_numberic_id[row[0]]}\t{row[1]}\t')
            str_genes=''
            for i in range(2,len(row)):
                str_genes=str_genes+row[i]+'@'+map_geneid_to_info[row[i]]['strand']+';'
            genepos_tsv.write(str_genes[:-1]+'\n')

def get_node_coverage(node_id):
    return float(node_id.split('_')[5])

def get_node_length(node_id):
    return float(node_id.split('_')[3])

def get_value(edge_df0, source_id, target_id):
    # if source_id in edge_df0['source'].values and target_id in edge_df0['target'].values:
    #     print(edge_df0[((edge_df0['source'] == source_id) & (edge_df0['target'] == target_id))])
    # else:
    #     print('No value')
    df_res = edge_df0[((edge_df0['source'] == source_id) & (edge_df0['target'] == target_id))]
    if len(df_res.index) >= 1:
        print(df_res.iloc[0, 2])
    else:
        df_res = edge_df0[((edge_df0['source'] == source_id+'_m0') & (edge_df0['target'] == target_id+'_m0'))]
        if len(df_res.index) >= 1:
            print(df_res.iloc[0, 2])
        else:
            print("No value")
# pangraph.edge_df0

def get_value_num(edge_df0, source_id, target_id):
    df_res = edge_df0[((edge_df0['source'] == source_id) & (edge_df0['target'] == target_id))]
    if len(df_res.index) >= 1:
        return (df_res.iloc[0, 2])
    else:
        df_res = edge_df0[((edge_df0['source'] == source_id+'_m0') & (edge_df0['target'] == target_id+'_m0'))]
        if len(df_res.index) >= 1:
            return (df_res.iloc[0, 2])
        else:
            return -1

def next_node_multi_1(target_contigs_list, idx):
    ## find the next node of multiplicity 1 after the index: idx
    current_idx = idx + 1
    while(get_node_coverage(target_contigs_list[current_idx]) >= 25.0):
        current_idx += 1
    return target_contigs_list[current_idx]

def next_node_length(target_contigs_list, idx, threshold=3000):
    ## find the next node of length >= threshold after the index: idx
    current_idx = idx + 1
    while(get_node_length(target_contigs_list[current_idx]) < threshold):
        current_idx += 1
    return target_contigs_list[current_idx]

def top_prev(prev, node):
    node_temp = prev[node]
    current_node = None
    while(node_temp != None):
        current_node = node_temp
        node_temp = prev[node_temp]
    return current_node

def recent_large_prev(prev, node, threshold):
    while(get_node_length(node) < threshold):
        node = prev[node]
        if node == None:
            return None 
    return node

def recent_large_after(after, node, threshold):
    while(get_node_length(node) < threshold):
        node = after[node]
        if node == None:
            return None 
    return node

def vote_sign(my_list):
    count = my_list.count('-')
    if 2*count >= len(my_list):
        return '-'
    else:
        return '+'

def binary_label(y_class):
    y_bin = []
    nonenan_index = []
    for i in range(len(y_class)):
        if y_class[i]=='R' or y_class[i]=='I':
            y_bin.append(1)
            nonenan_index.append(i)
        elif y_class[i]=='S':
            y_bin.append(0)
            nonenan_index.append(i)
        else:
            y_bin.append(y_class[i])
    return np.array(y_bin), nonenan_index

def parse_gff_AMRgene_finder(gff_fh, sample_id, min_protein_len=40):
    # gene_annotation = OrderedDict()
    # gene_position = OrderedDict()    
    # suffix = 1
    # bed_records = []
    # gene_index = 0
    seq_id = None
    min_cds_len = 3 * min_protein_len
    gene_list = []
    
    for line in gff_fh:            
        if line.startswith('##FASTA'):
            #Done reading gff, move on to reading fasta
            break

        if line[0] == '#':
            continue
        line = line.strip()
        #print(line)
        cells = line.split('\t')
        if cells[2] != 'CDS':
            continue
        if 'BARRGD' not in cells[8]:
            continue
        start = int(cells[3])
        end = int(cells[4])
        length = end - start + 1
        if length < min_cds_len:
            continue
        if length % 3 != 0:
            continue
        cells[0] = cells[0].replace('-','_') #make sure seq_id has no -
        
        if seq_id != cells[0]:
            seq_id = cells[0]
            gene_index = 0

        # strand = cells[6]
        tags = cells[8].split(';')
        gene_id = None
        gene_name = ''
        gene_product = ''
        for tag in tags:
            if tag.startswith('ID='):
                gene_id = tag[3:]
            elif tag.startswith('gene='):                    
                gene_name = tag[5:]
                gene_name = re.sub(r'\W', '_', gene_name)
            elif tag.startswith('product='):                    
                gene_product = tag[8:]
        if gene_id == None:
            continue

        # Ensure gene_id is in the format of sample_id-seq_id-gene_tag
        if not gene_id.startswith(sample_id + '-'):
            gene_id = sample_id + '-' + gene_id

        if not gene_id.startswith(sample_id + '-' + seq_id + '-'):
            gene_id = sample_id + '-' + seq_id + '-' + gene_id[len(sample_id)+1:]

        gene_list.append(gene_id)
    
    return gene_list

def parse_gff_AMRgene_finder_neighbors(gff_fh, sample_id, min_protein_len=40):
    # gene_annotation = OrderedDict()
    # gene_position = OrderedDict()    
    # suffix = 1
    # bed_records = []
    # gene_index = 0
    seq_id = None
    min_cds_len = 3 * min_protein_len
    
    tag_list = []
    gene_list = []
    for line in gff_fh:            
        if line.startswith('##FASTA'):
            #Done reading gff, move on to reading fasta
            break

        if line[0] == '#':
            continue
        line = line.strip()
        #print(line)
        cells = line.split('\t')
        if cells[2] != 'CDS':
            continue
        # if 'BARRGD' not in cells[8]:
        #     continue
        start = int(cells[3])
        end = int(cells[4])
        length = end - start + 1
        if length < min_cds_len:
            continue
        if length % 3 != 0:
            continue
        cells[0] = cells[0].replace('-','_') #make sure seq_id has no -
        
        if seq_id != cells[0]:
            seq_id = cells[0]
            gene_index = 0

        # strand = cells[6]
        tags = cells[8].split(';')
        gene_id = None
        gene_name = ''
        gene_product = ''
        for tag in tags:
            if tag.startswith('ID='):
                gene_id = tag[3:]
            elif tag.startswith('gene='):                    
                gene_name = tag[5:]
                gene_name = re.sub(r'\W', '_', gene_name)
            elif tag.startswith('product='):                    
                gene_product = tag[8:]
        if gene_id == None:
            continue

        # Ensure gene_id is in the format of sample_id-seq_id-gene_tag
        if not gene_id.startswith(sample_id + '-'):
            gene_id = sample_id + '-' + gene_id

        if not gene_id.startswith(sample_id + '-' + seq_id + '-'):
            gene_id = sample_id + '-' + seq_id + '-' + gene_id[len(sample_id)+1:]

        tag_list.append(cells[8])
        gene_list.append(gene_id)
    
    return tag_list, gene_list

def topfeatures():
    # Top correlation features
    pa_matrix = pd.read_csv(pantaout_dir+'gene_presence_absence.Rtab', sep='\t', index_col=0).T
    colums_feature = pa_matrix.columns
    sel = VarianceThreshold(threshold=0)
    pa_matrix = sel.fit_transform(pa_matrix)
    selected_columns = colums_feature[sel.get_support(indices=True)]
    metadata_panta = pd.read_csv("data/Ecoli1936metafiles/metadata_final.csv")
    from pangraph.utils import binary_label
    from sklearn.feature_selection import mutual_info_classif, chi2
    mutual_mat = []
    for idx in range(2, 14):
        y_class = metadata_panta.iloc[:,idx].values
        # print(metadata_panta.columns[idx])
        y, nonenan_index = binary_label(y_class) # v6
        pa_matrix_new = pa_matrix[nonenan_index, ]
        y_new = y[nonenan_index].astype(int)
        scores, pvalue = chi2(pa_matrix_new, y_new)
        mutual_mat.append(scores)
    mutual_mat = np.array(mutual_mat)
    mutual_mat_max = mutual_mat.max(axis=0)
    df_feature = pd.DataFrame({'gene': selected_columns, 'score': mutual_mat_max})
    df_feature = df_feature.sort_values(by='score', ascending=False)
    topfeaturesbypresenceabsence = df_feature['gene'][:400].values
    
    
    # ### Top correlation features from Sim Matrix
    pa_matrix = pd.read_csv(pantaout_dir+'gene_presence_absence.Rtab', sep='\t', index_col=0).T
    n_samples = pa_matrix.shape[0]
    n_genes = pa_matrix.shape[1]
    colsum = pa_matrix.sum()
    common_gene_cluster = [colsum.index[idx] for idx in range(n_genes) if colsum[idx] > 0.1*n_samples]
    simmatrix = np.load(pantaout_dir + 'similarsite.npy')
    metadata_panta = pd.read_csv("data/Ecoli1936metafiles/metadata_final.csv")
    from pangraph.utils import binary_label
    from sklearn.feature_selection import mutual_info_classif, chi2, f_classif

    mutual_mat = []
    for idx in range(2, 14):
        y_class = metadata_panta.iloc[:,idx].values
        # print(metadata_panta.columns[idx])
        y, nonenan_index = binary_label(y_class) # v6
        pa_matrix_new = simmatrix[nonenan_index, ]
        y_new = y[nonenan_index].astype(int)
        scores, pvalue = chi2(pa_matrix_new, y_new)
        mutual_mat.append(scores)
    mutual_mat = np.array(mutual_mat)
    mutual_mat_max = mutual_mat.max(axis=0)
    df_feature = pd.DataFrame({'gene': common_gene_cluster, 'score': mutual_mat_max})
    df_feature = df_feature.sort_values(by='score', ascending=False)
    topfeaturesbysimmat = df_feature['gene'][:400].values