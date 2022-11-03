import copy
import tqdm
import torch
import random
import constant
import numpy as np
import networkx as nx
import torch.nn as nn
from file_io import *
from graph import head_to_tree, tree_to_adj


def compute_centrality(ins, start, end):
    """
    Compute the average centrality of an entity.
    :param ins: certain instance (dict).
    :param start: start position (int).
    :param end:  end position (int).
    :return: the average centrality (float).
    """
    head_list = [int(i) for i in ins["stanford_head"]]
    root = head_to_tree(head_list)
    matrix = tree_to_adj(root, len(head_list), directed=False, self_loop=False)

    G = nx.Graph()
    nodes = range(len(matrix))
    G.add_nodes_from(nodes)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 1:
                G.add_edge(i, j)

    cc = nx.closeness_centrality(G)
    bc = nx.betweenness_centrality(G)
    dc = nx.degree_centrality(G)
    cc = np.array([cc[i] for i in range(len(head_list))])
    bc = np.array([bc[i] for i in range(len(head_list))])
    dc = np.array([dc[i] for i in range(len(head_list))])
    avgC = (cc + bc + dc)/3
    avgC = np.sum(avgC[start: end+1])
    return avgC


def match_entity(ins1_cen, ins2_cen, TDT):
    """
    Match entity pair whose TD value below to TDT.
    :param ins1_cen: centrality of instance 1 (list).
    :param ins2_cen: centrality of instance 2 (list).
    :param TDT: TD threshold (float).
    :return: matched entity pair (list).
    """
    entity_pair = []
    for i in range(len(ins1_cen)):
        for j in range(len(ins2_cen)):
            if abs(ins1_cen[i] - ins2_cen[j]) < TDT:
                entity_pair.append([i, j])
    if entity_pair:
        return entity_pair
    else:
        return None


def find_neighbor(head_list, start, end):
    """
    Find entity's neighbor.
    :param head_list: head indexes (list).
    :param start: start position (int).
    :param end: end position (int).
    :return: neighbor (list).
    """
    root = head_to_tree(head_list)
    matrix = tree_to_adj(root, len(head_list), directed=False, self_loop=False)

    neighbor = []
    for i in range(start, end+1):
        for j in range(len(matrix)):
            if matrix[i][j] == 1:
                neighbor.append(j)
    if neighbor:
        return neighbor
    else:
        return None


def compute_fs(ins1, ins2, pair):
    ins1_head = [int(i) for i in ins1["stanford_head"]]
    ins2_head = [int(i) for i in ins2["stanford_head"]]
    index2key = {0: "subj", 1: "obj"}
    ins1_nei = find_neighbor(ins1_head, ins1[index2key[pair[0]] + "_start"], ins1[index2key[pair[0]] + "_end"])
    ins2_nei = find_neighbor(ins2_head, ins2[index2key[pair[1]] + "_start"], ins2[index2key[pair[1]] + "_end"])
    if len(ins1_nei) == 0 or len(ins2_nei) == 0:
        return None

    pos_embs = nn.Embedding.from_pretrained(torch.load('emb/pos_emb_epoch_150.pt', map_location=torch.device('cpu')))
    dep_embs = nn.Embedding.from_pretrained(torch.load('emb/dep_emb_epoch_150.pt', map_location=torch.device('cpu')))
    pos_to_id = constant.POS_TO_ID
    dep_to_id = constant.DEPREL_TO_ID

    ins1_syn_list = []
    for nei in ins1_nei:
        nei_pos = ins1["stanford_pos"][nei]
        nei_dep = ins1["stanford_deprel"][nei]
        if nei_pos in pos_to_id.keys():
            nei_pos_emb = pos_embs(torch.tensor(pos_to_id[nei_pos]))
        else:
            nei_pos_emb = pos_embs(torch.tensor(pos_to_id["<UNK>"]))
        if nei_dep in dep_to_id.keys():
            nei_dep_emb = dep_embs(torch.tensor(dep_to_id[nei_dep]))
        else:
            nei_dep_emb = dep_embs(torch.tensor(dep_to_id["<UNK>"]))
        ins1_syn_list.append(torch.cat((nei_pos_emb, nei_dep_emb)))
    ins1_syn_emb = ins1_syn_list[0]
    for i in range(1, len(ins1_syn_list)):
        ins1_syn_emb += ins1_syn_list[i]

    ins2_syn_list = []
    for nei in ins2_nei:
        nei_pos = ins2["stanford_pos"][nei]
        nei_dep = ins2["stanford_deprel"][nei]
        if nei_pos in pos_to_id.keys():
            nei_pos_emb = pos_embs(torch.tensor(pos_to_id[nei_pos]))
        else:
            nei_pos_emb = pos_embs(torch.tensor(pos_to_id["<UNK>"]))
        if nei_dep in dep_to_id.keys():
            nei_dep_emb = dep_embs(torch.tensor(dep_to_id[nei_dep]))
        else:
            nei_dep_emb = dep_embs(torch.tensor(dep_to_id["<UNK>"]))
        ins2_syn_list.append(torch.cat((nei_pos_emb, nei_dep_emb)))
    ins2_syn_emb = ins2_syn_list[0]
    for i in range(1, len(ins2_syn_list)):
        ins2_syn_emb += ins2_syn_list[i]

    fs = torch.cosine_similarity(ins1_syn_emb, ins2_syn_emb, dim=0).item()
    return fs


def generate_cf(ins1, ins2, pair):
    index2key = {0: "subj", 1: "obj"}
    noun = ['NNS', 'NN', 'NNP', 'NNPS']
    verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adj = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'RP', 'WRB']
    ins1_head = [int(inx) for inx in ins1["stanford_head"]]
    ins2_head = [int(inx) for inx in ins2["stanford_head"]]

    record = {}
    for p in pair:
        ins1_nei = find_neighbor(ins1_head, ins1[index2key[p[0]] + "_start"], ins1[index2key[p[0]] + "_end"])
        ins2_nei = find_neighbor(ins2_head, ins2[index2key[p[1]] + "_start"], ins2[index2key[p[1]] + "_end"])
        if len(ins1_nei) == 0 or len(ins2_nei) == 0:
            continue
        else:
            for nei in ins1_nei:
                for can in ins2_nei:
                    nei_pos = ins1["stanford_pos"][nei]
                    if nei_pos in noun:
                        nei_pos = "noun"
                    elif nei_pos in verb:
                        nei_pos = "verb"
                    elif nei_pos in adj:
                        nei_pos = "adj"

                    can_pos = ins2["stanford_pos"][can]
                    if can_pos in noun:
                        can_pos = "noun"
                    elif can_pos in verb:
                        can_pos = "verb"
                    elif can_pos in adj:
                        can_pos = "adj"

                    if nei_pos == can_pos and ins1["token"][nei].lower() != ins2["token"][can].lower():
                        if nei in record.keys():
                            nei_cen = compute_centrality(ins1, nei, nei)
                            pre = record[nei]
                            pre_cen = compute_centrality(ins2, pre, pre)
                            can_cen = compute_centrality(ins2, can, can)
                            if abs(nei_cen - pre_cen) > abs(nei_cen - can_cen):
                                record[nei] = can
                        else:
                            record[nei] = can

    if record:
        ins_cf = copy.deepcopy(ins1)
        for key in record.keys():
            ins_cf["token"][key] = ins2["token"][record[key]]
            ins_cf["stanford_pos"][key] = ins2["stanford_pos"][record[key]]
        ins_cf["relation"] = ins2["relation"]
        ins_cf["origin"] = ins1["id"]
        assert len(ins_cf["token"]) == len(ins1["token"])
        assert ins_cf["token"] != ins1["token"]
        return ins_cf
    else:
        return None


if __name__ == "__main__":
    TDT = 0.2
    FST = 0.7

    file = "data/train.json"
    ins = read_json_file(file)
    ins_cf = []
    for i in tqdm.tqdm(ins):
        ins1 = i
        sample_num = 0
        ins2_list = []
        ins2_list.append(ins1)
        while sample_num < 3:
            ins2 = random.choice(ins)
            if ins2 not in ins2_list:
                sample_num += 1
                ins2_list.append(ins2)

                if ins1["relation"] != ins2["relation"]:
                    ins1_sub = compute_centrality(ins1, ins1["subj_start"], ins1["subj_end"])
                    ins1_obj = compute_centrality(ins1, ins1["obj_start"], ins1["obj_end"])
                    ins2_sub = compute_centrality(ins2, ins2["subj_start"], ins2["subj_end"])
                    ins2_obj = compute_centrality(ins2, ins2["obj_start"], ins2["obj_end"])
                    pair = match_entity([ins1_sub, ins1_obj], [ins2_sub, ins2_obj], TDT)

                    if pair:
                        valid_pair = []
                        for p in pair:
                            fs = compute_fs(ins1, ins2, p)
                            if fs >= FST:
                                valid_pair.append(p)
                        if valid_pair:
                            cf = generate_cf(ins1, ins2, valid_pair)
                            if cf:
                                ins_cf.append(cf)

    output = "data/cf.json"
    write_json_file(output, ins_cf)
