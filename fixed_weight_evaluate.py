# input:test_group_event
#       group_users
#       user_emb,luser_emb,ruser_emb,item_emb
# output:hit@n, MRR
import pandas as pd
import numpy as np
import math
from file_process import get_group_users

# test_group_event_file = "./data/dataset/kaggle/test_groupid_eventid_candis.dat"
# group_user_file = "./data/dataset/kaggle/test_groupid_userids.dat"
# user_emb_file = "./data/vectors/kaggle/user100000"
# item_emb_file = "./data/vectors/kaggle/item100000"
# luser_emb_file = "./data/vectors/kaggle/luser100000"
# ruser_emb_file = "./data/vectors/kaggle/ruser100000"
# yelp our model
# test_group_event_file = "./data/dataset/less_50/douban_sh/test_groupid_eventid_candis.dat"
# group_user_file = "./data/dataset/less_50/douban_sh/test_groupid_userids.dat"
# user_emb_file = "./data/vectors/less_50/douban_sh/fixed_weight_model/joint/user500000"
# item_emb_file = "./data/vectors/less_50/douban_sh/fixed_weight_model/joint/item500000"
# user_weight_map_file = "./data/vectors/less_50/douban_sh/fixed_weight_model/joint/user_weight500000"

test_group_event_file = "./data/dataset/less_50/yelp/test_groupid_eventid_candis.dat"
group_user_file = "./data/dataset/less_50/yelp/test_groupid_userids.dat"
user_emb_file = "./data/vectors/less_50/yelp/our_model/joint/user_init"
item_emb_file = "./data/vectors/less_50/yelp/our_model/joint/item_init"
# user_emb_file = "./data/baseline/yelp/vectors/user7361440"
# item_emb_file = "./data/baseline/yelp/vectors/item7361440"
user_weight_map_file = "./data/vectors/less_50/yelp/our_model/joint/user_weight_init"
DIM = 50


def get_emb(vertex_emb_file):
    df = pd.read_csv(vertex_emb_file, sep="\t", names=["vertex", "emb"], engine="python")
    vertex_emb = dict()
    for index, row in df.iterrows():
        vertex_emb[row["vertex"]] = np.array(str(row["emb"]).strip().split(" ")).astype(np.float32)
    return vertex_emb


def get_user_weight(user_weight_file):
    user_weight_map = dict()
    df = pd.read_csv(user_weight_file,sep="\t",names=["user","weight"],engine="python")
    for index,row in df.iterrows():
        user_weight_map[row["user"]] = row["weight"]
    return user_weight_map

user_emb = get_emb(user_emb_file)
item_emb = get_emb(item_emb_file)
user_weight_map = get_user_weight(user_weight_map_file)

group_users = get_group_users(group_user_file)


def cal_group_emb(members):
    group_emb = np.zeros(DIM,)
    #print(user_weight_map)
    total_weight = 0.0
    for member in members:
        total_weight += user_weight_map.get(member)
        group_emb += user_weight_map.get(member)*user_emb.get(member)
    group_emb /= total_weight
    return group_emb


# calculate topk hit ratio and MRR
def cal_topk_list(group, event, candi_list, k):
    members = group_users.get(group)
    group_emb = cal_group_emb(members)
    rec_events_dict = dict()
    rec_events_dict[event] = group_emb.dot(item_emb.get(event))
    for candi in candi_list:
        rec_events_dict[candi] = group_emb.dot(item_emb.get(candi))
    # sort recommendation list
    sorted_rec_events_dict = sorted(rec_events_dict.items(), key=lambda d: d[1], reverse=True)
    rank = hit = rr = 0
    for t in sorted_rec_events_dict:
        rank += 1
        if event == t[0]:
            if rank <= k:
                hit = 1
            rr = 1 / float(rank)
            break
    return hit, rr


if __name__ == "__main__":
    top_k = [1,5,10,15,20,40,60,80,100]
    for k in top_k:
        df = pd.read_csv(test_group_event_file, sep="\t", names=["group", "event", "neg_candi"], engine="python")
        avg_hit = MRR = 0.0
        for index, row in df.iterrows():
            candi_list = [int(item) for item in str(row["neg_candi"]).strip().split(" ")]
            hit, rr = cal_topk_list(row["group"], row["event"], candi_list, k)
            avg_hit += hit
            MRR += rr
        print("The avg hit ratio@%d is :%f,MRR is %f" % (k, avg_hit / len(df), MRR / len(df)))
