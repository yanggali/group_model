import pandas as pd
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
    # for i in col1:
    #     # ver1_num[i] = len(df[df[col_names[0]]==i][col_names[1]])
    #     ver1_neighbours[i] = list(df[df[col_names[0]]==i][col_names[1]])
    #     ver1_num[i] = len(ver1_neighbours[i])
    # for j in col2:
    #     # ver2_num[j] = len(df[df[col_names[1]]==j][col_names[0]])
    #     ver2_neighbours[j] = list(df[df[col_names[1]]==j][col_names[0]])
    #     ver2_num[i] = len(ver2_neighbours[j])
    return ver1_num,ver2_num,ver1_neighbours,ver2_neighbours,all_tuples


def get_group_users(inputfile,col_names=["groupid","users"],sep="\t"):
    df = pd.read_csv(inputfile,sep=sep,names=col_names,engine="python")
    group_users = dict()
    for index,row in df.iterrows():
        group_users[row["groupid"]] = [int(member) for member in list(str(row["users"]).strip().split(" "))]
    return group_users


def write_to_file(filename,ver_emb):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename,'a') as fw:
        for k,v in ver_emb.items():
            write_str = str(k)+"\t"
            for e in v:
                write_str+=str(e)+" "
            write_str = write_str.strip()+"\n"
            fw.write(write_str)




