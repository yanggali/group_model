import pandas as pd
import numpy as np
import os


def read_file(inputfile,col_names,sep="\t"):
    df = pd.read_csv(inputfile,sep=sep,names=col_names,engine="python")
    all_tuples = list()
    # col1 = set(df[col_names[0]])
    # col2 = set(df[col_names[1]])
    ver1_num = dict()
    ver2_num = dict()
    ver1_neighbours = dict()
    ver2_neighbours = dict()
    for index,row in df.iterrows():
        all_tuples.append((row[col_names[0]],row[col_names[1]]))
        if row[col_names[0]] not in ver1_neighbours:
            ver1_neighbours[row[col_names[0]]] = [row[col_names[1]]]
            ver1_num[row[col_names[0]]] = 1
        else:
            ver1_neighbours[row[col_names[0]]].append(row[col_names[1]])
            ver1_num[row[col_names[0]]] += 1
        if row[col_names[1]] not in ver2_neighbours:
            ver2_neighbours[row[col_names[1]]] = [row[col_names[0]]]
            ver2_num[row[col_names[1]]] = 1
        else:
            ver2_neighbours[row[col_names[1]]].append(row[col_names[0]])
            ver2_num[row[col_names[1]]] += 1
    return ver1_num,ver2_num,ver1_neighbours,ver2_neighbours,all_tuples


def get_group_users(inputfile,col_names=["groupid","users"],sep="\t"):
    df = pd.read_csv(inputfile,sep=sep,names=col_names,engine="python")
    group_users = dict()
    for index,row in df.iterrows():
        group_users[row["groupid"]] = [int(member) for member in list(str(row["users"]).strip().split(" "))]
    return group_users


def get_user_groups(inputfile,col_names=["groupid","users"],sep="\t"):
    df = pd.read_csv(inputfile,sep=sep,names=col_names,engine="python")
    user_groups = dict()
    user_groups_degree = dict()
    for index,row in df.iterrows():
        users = [int(user) for user in str(row["users"]).split(" ")]
        for user in users:
            if user not in user_groups:
                user_groups[user] = set()
                user_groups_degree[user] = 0
            user_groups[user].add(row["groupid"])
            user_groups_degree[user] += 1
    return user_groups,user_groups_degree


def get_emb(vertex_emb_file,vertex_emb):
    df = pd.read_csv(vertex_emb_file, sep="\t", names=["vertex", "emb"], engine="python")
    for index, row in df.iterrows():
        vertex_emb[row["vertex"]] = np.array(str(row["emb"]).strip().split(" ")).astype(np.float32)


def emb_to_file(filename,ver_emb):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename,'w') as fw:
        for k,v in ver_emb.items():
            write_str = str(k)+"\t"
            for e in v:
                write_str+=str(e)+" "
            write_str = write_str.strip()+"\n"
            fw.write(write_str)


def userweight_to_file(filename,user_weight):
    if os.path.exists(filename):
        os.remove(filename)
    user_weight_str = ""
    with open(filename, 'a') as fw:
        for user,weight in user_weight.items():
            user_weight_str += str(user)+"\t"+str(weight)+"\n"
        fw.write(user_weight_str)


def featureweight_to_file(filename,feature_weight):
    if os.path.exists(filename):
        os.remove(filename)
    feature_weight_str = ""
    with open(filename, 'a') as fw:
        for feature,weight in feature_weight.items():
            feature_weight_str += str(feature)+"\t"+str(weight)+"\n"
        fw.write(feature_weight_str)

def read_int_to_list(filename):
    read_list = list()
    read_str = ""
    with open(filename,'r') as fr:
        read_str = fr.readline()

    read_list = [int(node) for node in read_str.strip().split(" ")]
    return read_list

def read_float_to_list(filename):
    read_list = list()
    read_str = ""
    with open(filename, 'r') as fr:
        read_str = fr.readline().strip()
    read_list = [float(node) for node in read_str.strip().split(" ")]
    return read_list


def read_to_dict(filename):
    read_dict = dict()
    with open(filename,'r') as fr:
        for line in fr.readlines():
            line_str = line.split("\t")
            read_dict[int(line_str[0])] = [float(node) for node in line_str[1].split(" ")]
    return read_dict


def write_to_file(filename,str):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename,'w') as fw:
        fw.write(str)


def append_to_file(filename,str):
    with open(filename,'a') as fw:
        fw.write(str)


def read_to_map(filename,sep="\t",col_names=["col1","col2"]):
    read_map = dict()
    df = pd.read_csv(filename,sep=sep,names=col_names,engine="python")
    for index,row in df.iterrows():
        read_map[row[col_names[0]]] = row[col_names[1]]
    return  read_map


