import copy
import tqdm
import torch
import random
import numpy as np
import networkx as nx
from file_io import *
from supar import Parser
from load_glove import load_glove
from stanfordcorenlp import StanfordCoreNLP


def generate_sd_mat(token, sd_parser, nlp_parser):
    sentence = " ".join(token)
    pos = nlp_parser.pos_tag(sentence)
    sd_input = [[(item[0], item[0], item[1]) for item in pos]]
    sd_result = sd_parser.predict(sd_input, verbose=False)[0].values[8]
    try:
        assert len(token) == len(sd_result)
    except AssertionError:
        return None
    sd_mat = np.zeros((len(token), len(token)), dtype=np.float32)
    for i, item in enumerate(sd_result):
        if "|" in item:
            heads = [int(h.split(":")[0])-1 for h in item.split("|")]
            for head in heads:
                sd_mat[i][head] = 1
                sd_mat[head][i] = 1
        elif item != "_":
            if "root" not in item:
                head = int(item.split(":")[0])-1
                sd_mat[i][head] = 1
                sd_mat[head][i] = 1
    return sd_mat


def compute_sp_path(mat, start, end):
    G = nx.Graph()
    nodes = range(len(mat))
    G.add_nodes_from(nodes)
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i][j] == 1:
                G.add_edge(i, j)
    try:
        sp = nx.shortest_path(G, source=start, target=end)
        return sp
    except nx.exception.NetworkXNoPath:
        return None


def compute_ss(glove, tok1, sp1, tok2, sp2, dim=300):
    sp_tok1 = [tok1[i] for i in sp1]
    sp_tok2 = [tok2[i] for i in sp2]
    sp1_vec = np.zeros(dim)
    for w in sp_tok1:
        if w in glove.keys():
            sp1_vec += glove[w]
    sp1_vec = torch.tensor(sp1_vec/len(sp1))

    sp2_vec = np.zeros(dim)
    for w in sp_tok2:
        if w in glove.keys():
            sp2_vec += glove[w]
    sp2_vec = torch.tensor(sp2_vec/len(sp2))

    ss = torch.cosine_similarity(sp1_vec, sp2_vec, dim=0).item()
    return ss


def filter_between(ins, path):
    ent = [ins["subj_start"], ins["subj_end"], ins["obj_start"], ins["obj_end"]]
    bwt = []
    for i in path:
        if i not in ent:
            bwt.append(i)
    return bwt


def parse_pos(pos_list):
    pos = []
    for item in pos_list:
        tag = item[1]
        if ":" in tag:
            pos.append(tag.split(":")[0])
        else:
            pos.append(tag)
    return pos


def parse_depen(parse_list):
    temp = {}
    for item in parse_list:
        temp[item[2]] = [str(item[1]), item[0]]
    stanford_head = []
    stanford_deprel = []
    for i in range(1, len(parse_list)+1):
        stanford_head.append(temp[i][0])
        stanford_deprel.append(temp[i][1])
    return stanford_head, stanford_deprel


def do_exchange(nlp_parser, ins1, bwt1, ins2, bwt2):
    cf_ins = copy.deepcopy(ins1)
    cf_ins["token"] = ins1["token"][:bwt1[0]]
    cf_ins["token"].extend(ins2["token"][bwt2[0]: bwt2[-1]+1])
    cf_ins["token"].extend(ins1["token"][bwt1[-1]+1:])
    if len(cf_ins["token"]) != len(ins1["token"]):
        return None
    elif cf_ins["token"] == ins1["token"]:
        return None
    else:
        sent = " ".join(cf_ins["token"])
        pos_result = nlp_parser.pos_tag(sent)
        stanford_pos = parse_pos(pos_result)
        cf_ins["stanford_pos"] = stanford_pos
        depen_result = nlp_parser.dependency_parse(sent)
        stanford_head, stanford_deprel = parse_depen(depen_result)
        cf_ins["stanford_head"] = stanford_head
        cf_ins["stanford_deprel"] = stanford_deprel
        cf_ins["relation"] = ins2["relation"]
        cf_ins["origin"] = ins1["id"]
        cf_ins["combination"] = ins2["id"]
        return cf_ins


if __name__ == "__main__":
    SST = 0.6

    sd_parser = Parser.load("biaffine-sdp-en")
    nlp_parser = StanfordCoreNLP("../library/stanford-corenlp-4.4.0")
    glove = load_glove()

    file_input = "data/train.json"
    file_output = "data/cf.json"
    ins = read_json_file(file_input)
    ins_cfs = []
    for ins1 in tqdm.tqdm(ins):
        sample_num = 0
        record = [ins1]
        while sample_num < 3:
            ins2 = random.choice(ins)
            if ins2 not in record:
                sample_num += 1
                record.append(ins2)

                if ins1["relation"] != ins2["relation"]:
                    ins1_mat = generate_sd_mat(ins1["token"], sd_parser, nlp_parser)
                    ins2_mat = generate_sd_mat(ins2["token"], sd_parser, nlp_parser)
                    if ins1_mat is not None and ins2_mat is not None:
                        ins1_sp = compute_sp_path(ins1_mat, ins1["subj_start"], ins1["obj_end"])
                        ins2_sp = compute_sp_path(ins2_mat, ins2["subj_start"], ins2["obj_end"])
                        if ins1_sp is not None and ins2_sp is not None:
                            ss = compute_ss(glove, ins1["token"], ins1_sp, ins2["token"], ins2_sp)
                            if ss >= SST:
                                ins1_bwt = filter_between(ins1, ins1_sp)
                                ins2_bwt = filter_between(ins2, ins2_sp)
                                if ins1_bwt and ins2_bwt and len(ins1_bwt) == len(ins2_bwt):
                                    ins_cf = do_exchange(nlp_parser, ins1, ins1_bwt, ins2, ins2_bwt)
                                    if ins_cf is not None:
                                        ins_cfs.append(ins_cf)

    write_json_file(file_output, ins_cfs)
