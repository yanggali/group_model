# calculate users' social features and attendness features
from file_process import read_file,get_user_groups,read_to_map
import pickle
import json
inputdata1 = "./data/dataset/less_50/douban_bj/train_userid_eventid.dat"
inputdata3 = "./data/dataset/less_50/douban_bj/train_groupid_userids.dat"
user_pagerank = "./data/dataset/less_50/douban_bj/user_pagerank.dat"
user_feature_output = "./data/dataset/less_50/douban_bj/user_features.dat"
user_feature_map = dict()

user_degree, itematuser_degree, user_nei, itematuser_nei, all_edges_1 = read_file(inputdata1, ["userid", "eventid"])
user_groups,user_groups_degree = get_user_groups(inputdata3, ["groupid", "users"])
max_event_attend = max(user_degree.values())
max_group_attend = max(user_groups_degree.values())
user_pagerank_map = read_to_map(user_pagerank)
max_pagerank = max(user_pagerank_map.values())
for user,group_num in user_groups_degree.items():
    if user not in user_feature_map:
        user_feature_map[user] = dict()
    user_feature_map[user]["event_attend"] = user_degree.get(user)/max_event_attend
    user_feature_map[user]["group_attend"] = group_num/max_group_attend
    user_feature_map[user]["pagerank"] = user_pagerank_map.get(user)/max_pagerank

user_feature_json = json.dumps(user_feature_map)

output = open(user_feature_output,"w")
output.write(user_feature_json)
output.close()