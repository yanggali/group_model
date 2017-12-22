# input:test_group_event
#       group_users
#       user_emb,luser_emb,ruser_emb,item_emb
# output:hit@n, MRR
import pandas as pd
import numpy as np
import math
from file_process import get_group_users
# test_group_event_file = "./data/dataset/kaggle/test_groupid_event_candis.dat"
# group_user_file = "./data/dataset/kaggle/test_groupid_users.dat"
# user_emb_file = "./data/vectors/kaggle/user485676"
# item_emb_file = "./data/vectors/kaggle/item485676"
# luser_emb_file = "./data/vectors/kaggle/luser485676"
# ruser_emb_file = "./data/vectors/kaggle/ruser485676"
# plancast
test_group_event_file = "./data/dataset/plancast/test_groupid_eventid_candis.dat"
group_user_file = "./data/dataset/plancast/test_groupid_userids.dat"
user_emb_file = "./data/vectors/plancast/user1000000"
item_emb_file = "./data/vectors/plancast/item1000000"
luser_emb_file = "./data/vectors/plancast/luser1000000"
ruser_emb_file = "./data/vectors/plancast/ruser1000000"
DIM = 50


def get_emb(vertex_emb_file):
    df = pd.read_csv(vertex_emb_file,sep="\t",names=["vertex","emb"],engine="python")
    vertex_emb = dict()
    for index,row in df.iterrows():
        vertex_emb[row["vertex"]] = np.array(str(row["emb"]).strip().split(" ")).astype(np.float32)
    return vertex_emb


user_emb = get_emb(user_emb_file)
item_emb = get_emb(item_emb_file)
luser_emb = get_emb(luser_emb_file)
ruser_emb = get_emb(ruser_emb_file)
group_users = get_group_users(group_user_file)


def cal_group_emb(members):
    sum_weight = 0
    member_weight = dict()
    group_emb = np.zeros(DIM,)
    for i in members:
        member_weight[i] = 0
        member_emb = user_emb.get(i)
        for j in members:
            if j != i:
                w_ij = math.exp(luser_emb.get(i).dot(ruser_emb.get(j)))
                member_weight[i] += w_ij
                sum_weight += w_ij
        group_emb += member_weight[i] * member_emb
    group_emb /= sum_weight
    return group_emb


# calculate topk hit ratio and MRR
def cal_topk_list(group,event,candi_list,k):
    members = group_users.get(group)
    group_emb = cal_group_emb(members)
    rec_events_dict = dict()
    rec_events_dict[event] = group_emb.dot(item_emb.get(event))
    for candi in candi_list:
        rec_events_dict[candi] = group_emb.dot(item_emb.get(candi))
    # sort recommendation list
    sorted_rec_events_dict = sorted(rec_events_dict.items(), key=lambda d: d[1],reverse=True)
    rank = hit = rr = 0
    for t in sorted_rec_events_dict:
        rank += 1
        if event == t[0]:
            if rank <= k:
                hit = 1
            rr = 1/float(rank)
            break
    return hit,rr


if __name__=="__main__":
    top_k = [20,15,10,5,1]
    for k in top_k:
        df = pd.read_csv(test_group_event_file,sep="\t",names=["group","event","neg_candi"],engine="python")
        avg_hit = MRR = 0.0
        for index,row in df.iterrows():
            candi_list = [ int(item) for item in str(row["neg_candi"]).strip().split(" ")]
            hit,rr = cal_topk_list(row["group"],row["event"],candi_list,k)
            avg_hit += hit
            MRR += rr
        print("The avg hit ratio@%d is :%f,MRR is %f" % (k,avg_hit/len(df),MRR/len(df)))