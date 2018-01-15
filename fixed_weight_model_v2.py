from file_process import read_file, get_group_users, emb_to_file, \
    userweight_to_file,get_emb,featureweight_to_file,write_to_file,append_to_file
import numpy as np
import random
import math
import json

DIM = 50
NEG_N = 2  # number of negative samples
init_lr = 0.025
lr = init_lr
ita = 0.5
gama = 1
lower_bound = 0.1
reg = 0.1
# vertices and their degrees
user_degree = dict()
itematuser_degree = dict()
group_degree = dict()
itematgroup_degree = dict()

# all edges
all_edges_1 = list()
all_edges_2 = list()

# all vertices set
all_v = set()
all_user_v = set()
all_item_v = set()
all_luser_v = set()
all_ruser_v = set()

# vertex sample table
pop_itematuser_sample = list()
pop_itematgroup_sample_dict = dict()
pop_itematgroup_sample = list()
# neg vertex sample table
itematuser_neg_table = list()
itematgroup_neg_table = list()
neg_table_size = int(1e8)
neg_sampling_power = 0.75

# vertex and its neighbours
user_nei = dict()
itematuser_nei = dict()
group_nei = dict()
itematgroup_nei = dict()
group_users = dict()

# all vertices' embedding
user_emb = dict()
item_emb = dict()
user_weight_map = dict()
# get feature weight
social_features = ["pagerank","event_attend","group_attend"]
feature_weight = dict()
user_feature_weight = dict()

SIGMOID_BOUND = 6
sigmoid_table_size = 1000
sig_map = dict()
# group_sample_file = "./data/dataset/less_50/yelp/group_sample.dat"
# inputdata1 = "./data/dataset/less_50/yelp/train_userid_eventid.dat"
# inputdata2 = "./data/dataset/less_50/yelp/train_groupid_eventid.dat"
# inputdata3 = "./data/dataset/less_50/yelp/train_groupid_userids.dat"
# inputdate4 = "./data/dataset/less_50/yelp/user_features.dat"
# init_user = "./data/baseline/yelp/vectors/user460090"
# init_item = "./data/baseline/yelp/vectors/item460090"
# out_user = "./data/vectors/less_50/yelp/our_model/stage2/user"
# out_item = "data/vectors/less_50/yelp/our_model/stage2/item"
# out_user_weight = "data/vectors/less_50/yelp/our_model/stage2/user_weight"
# out_feature_weight = "data/vectors/less_50/yelp/our_model/stage2/feature_weight"

inputdata1 = "./data/dataset/less_50/yelp/train_userid_eventid.dat"
inputdata2 = "./data/dataset/less_50/yelp/train_groupid_eventid.dat"
inputdata3 = "./data/dataset/less_50/yelp/train_groupid_userids.dat"
inputdate4 = "./data/dataset/less_50/yelp/user_features.dat"
init_user = "./data/baseline/yelp/vectors/user7361440"
init_item = "./data/baseline/yelp/vectors/item7361440"
out_user = "./data/vectors/less_50/yelp/our_model/stage2/user"
out_item = "data/vectors/less_50/yelp/our_model/stage2/item"
out_user_weight = "data/vectors/less_50/yelp/our_model/stage2/user_weight"
out_feature_weight = "data/vectors/less_50/yelp/our_model/stage2/feature_weight"


# initialize all edges, all vertices' degree and all vertices' neighbours
user_degree, itematuser_degree, user_nei, itematuser_nei, all_edges_1 = read_file(inputdata1, ["userid", "eventid"])

# get user feature weight
user_event_N = len(all_edges_1)
print("user event N is %d" % user_event_N)
print("user-event finished.")
group_degree, itematgroup_degree, group_nei, itematgroup_nei, all_edges_2 = read_file(inputdata2,
                                                                                      ["groupid", "eventid"])
group_event_N = len(all_edges_2)
print("group event N is %d" % group_event_N)
print("group-event finished.")
group_users = get_group_users(inputdata3, ["groupid", "users"])
print("user set:%d ,item set:%d, group set:%d" % (
    len(user_degree.keys()), len(itematuser_degree.keys()), len(group_degree.keys())))
user_list = list(user_degree.keys())
itematuser_list = list(itematuser_degree.keys())
group_list = list(group_degree.keys())
itematgroup_list = list(itematgroup_degree.keys())
print("item size:%d" % len(itematuser_list))
# get user feature map function
with open(inputdate4,'r') as load_f:
    user_feature_weight = json.load(load_f)

# initial negative sampling table
def init_vertex_neg_table(vertex_neg_table, vertex_degree, vertex_list):
    sum = cur_num = por = 0.0
    vid = 0
    for vertex, degree in vertex_degree.items():
        sum += math.pow(degree, neg_sampling_power)
    for k in range(neg_table_size):
        if float(k + 1) / neg_table_size > por:
            cur_num += math.pow(vertex_degree.get(vertex_list[vid]), neg_sampling_power)
            por = cur_num / sum
            vid += 1
        vertex_neg_table.append(vertex_list[vid - 1])


# initial neg sampling table
def init_neg_table():
    init_vertex_neg_table(itematgroup_neg_table, itematgroup_degree, itematgroup_list)

    # for user-event using negative sampling table
    # init_vertex_neg_table(itematuser_neg_table,itematuser_degree,itematuser_list)
    # for group-event using common table
    cal_pop_pro_in_group(itematgroup_list, itematuser_nei, pop_itematgroup_sample_dict)


def cal_pop_pro(vertex_list, vertex_degree, sample_list):
    totalweight = 0.0
    for vertex, degree in vertex_degree.items():
        totalweight += math.pow(degree, neg_sampling_power)
    sample_list.append(float(math.pow(vertex_degree[vertex_list[0]], neg_sampling_power) / totalweight))
    for vertex in vertex_list[1:]:
        sample_list.append(sample_list[-1] + math.pow(float(vertex_degree[vertex]), neg_sampling_power) / totalweight)
    sample_list[-1] = 1.0


def cal_pop_pro_in_group(vertex_list, vertex_nei, group_sample_list_dict):
    index = 0
    for group, members in group_users.items():
        group_sample_list_dict[group] = list()
        sum_degree = 0.0
        group_sample_list_dict[group].append(0)
        for item in vertex_list:
            degree_sum = 0.0
            for member in members:
                if member in vertex_nei[item]:
                    degree_sum += 1
            group_sample_list_dict[group].append(group_sample_list_dict[group][-1] +
                                                 math.pow(degree_sum + gama, neg_sampling_power))
            sum_degree += math.pow(degree_sum + gama, neg_sampling_power)
        del group_sample_list_dict[group][0]
        for i in range(len(vertex_list)):
            group_sample_list_dict[group][i] /= sum_degree
        index += 1
        if index%10000 == 0:
            print("%d row's group finished" % index)
    # for group, members in group_users.items():
    #     group_sample_list_dict[group] = list()
    #     for item in vertex_list:
    #         degree_sum = 0.0
    #         for member in members:
    #             if member in vertex_nei[item]:
    #                 degree_sum += 1
    #         group_sample_list_dict[group].append(math.pow(degree_sum + gama, neg_sampling_power))
    # for group in group_sample_list_dict.keys():
    #     group_sample_list = group_sample_list_dict.get(group)
    #     totalweight = sum(group_sample_list)
    #     group_sample_list[0] /= totalweight
    #     for k in range(1, len(group_sample_list)):
    #         group_sample_list[k] = group_sample_list[k - 1] + group_sample_list[k] / totalweight
    #     group_sample_list[-1] = 1.0
    #     group_sample_list_dict[group] = group_sample_list


def pop_itematgroup_sample_dict_to_file(vertex_list, vertex_nei, group_sample_list_dict):
    pop_itematgroup_sample_dict_str = ""
    index = 0
    for group, members in group_users.items():
        if index%1000 == 0:
            append_to_file(group_sample_file,pop_itematgroup_sample_dict_str)
            pop_itematgroup_sample_dict_str = ""
        group_sample_list_dict[group] = list()
        sum_degree = 0.0
        group_sample_list_dict[group].append(0)
        for item in vertex_list:
            degree_sum = 0.0
            for member in members:
                if member in vertex_nei[item]:
                    degree_sum += 1
            group_sample_list_dict[group].append(group_sample_list_dict[group][-1]+
                                                 math.pow(degree_sum + gama, neg_sampling_power))
            sum_degree += math.pow(degree_sum + gama, neg_sampling_power)
        del group_sample_list_dict[group][0]
        sample_list_str = ""
        for i in range(len(vertex_list)):
            group_sample_list_dict[group][i] /= sum_degree
            sample_list_str += str(group_sample_list_dict[group][i])+" "
        pop_itematgroup_sample_dict_str += str(group) +"\t"+sample_list_str.strip()+"\n"
        index += 1
        print("%d row's group finished" % index)
    append_to_file(group_sample_file,pop_itematgroup_sample_dict_str)


# initial sample table
def get_pop_pro():
    cal_pop_pro(itematuser_list, itematuser_degree, pop_itematuser_sample)
    print("user event sample finished")
    cal_pop_pro(itematgroup_list, itematgroup_degree, pop_itematgroup_sample)
    #cal_pop_pro_in_group(itematgroup_list, itematuser_nei, pop_itematgroup_sample_dict)
    print("group event sample finished")


def gen_gaussian():
    max_value = 32767
    vector = np.zeros(DIM, )
    for i in range(DIM):
        vector[i] = (random.randint(0, max_value) * 1.0 / max_value - 0.5) / DIM
    return vector


def init_vec(vec_list, vec_emb_dict):
    for vec in vec_list:
        vec_emb_dict[vec] = gen_gaussian()


def math_exp(x):
    try:
        ans = math.exp(x)
        # ans = math.log(x)
    except OverflowError:
        ans = float('inf')
        # ans = 1.79769313e+308
    return ans


def init_user_weight(users_at_group, user_weight):
    user_len = len(users_at_group)
    max_value = 32767
    for user in users_at_group:
        user_weight[user] = (random.randint(0, max_value) * 1.0 / max_value - 0.5) / user_len


def init_feature_weight(feature_weight):
    l = len(social_features)
    for feature in social_features:
        feature_weight[feature] = 1.0/l


# initial vertices' embedding
def init_all_vec():
    init_vec(user_list, user_emb)
    init_vec(itematuser_list, item_emb)
    useratgroup_set = set()
    for v in group_users.values():
        useratgroup_set.update(v)
    init_user_weight(useratgroup_set, user_weight_map)
    init_feature_weight(feature_weight)
    emb_to_file(out_user+"_init", user_emb)
    emb_to_file(out_item+"_init", item_emb)
    userweight_to_file(out_user_weight+"_init",user_weight_map)
    featureweight_to_file(out_feature_weight+"_init",feature_weight)


def init_vec_fromfile(emb_file, vertice_emb):
    vertice_emb = get_emb(emb_file)


def init_sigmod_table():
    for k in range(sigmoid_table_size):
        x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND
        sig_map[k] = 1 / (1 + math_exp(-x))


# sample an edge randomly
def draw_tuple(tuple_list):
    return tuple_list[random.randint(0, len(tuple_list) - 1)]


# sample vertex
def draw_vertex(vertex_list, vertex_sample_pro):
    r = random.random()
    for i in range(len(vertex_sample_pro)):
        if vertex_sample_pro[i] > r:
            break
    return vertex_list[i]


def sigmoid(x):
    if x > SIGMOID_BOUND: return 1
    if x < -SIGMOID_BOUND: return 0
    k = int((x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2)
    return sig_map.get(k)


# update user-item target vertex
def update_user_item_vertex(source, target, error, label):
    source_emb = user_emb.get(source)
    target_emb = item_emb.get(target)
    score = sigmoid(math.pow(-1, label) * source_emb.dot(target_emb))
    g = math.pow(-1, label) * score * lr
    # total error for source vertex
    error += g * target_emb
    new_vec = target_emb - g * source_emb
    item_emb[target] = new_vec


# calculate groups' embedding
def cal_group_emb(members):
    group_emb = np.zeros(DIM, )
    # print(user_weight_map)
    total_weight = 0.0
    for member in members:
        total_weight += user_weight_map.get(member)
        group_emb += user_weight_map.get(member) * user_emb.get(member)
    group_emb = np.divide(group_emb,total_weight)
    return group_emb, total_weight


# update vertices according to an edge
def update_user_item(source, target, neg_vertices):
    error = np.zeros(DIM, )
    update_user_item_vertex(source, target, error, 1)
    M = len(neg_vertices)
    if M != 0:
        for i in range(M):
            update_user_item_vertex(source, neg_vertices[i], error, 0)
    new_vector = user_emb.get(source) - error
    user_emb[source] = new_vector


def update_item_vertex_in_group(source_emb, target, target_emb, error, label):
    if math.isnan(source_emb.dot(target_emb)):
        print("not a number")
    score = sigmoid(math.pow(-1, label) * source_emb.dot(target_emb))
    g = math.pow(-1, label) * score * lr
    new_vec = target_emb - g * source_emb
    item_emb[target] = new_vec
    # total error for puser , luser and ruser vertex
    error += g * target_emb


def update_item_vertex_in_group_stage1(source_emb, target_emb, error, label):
    if math.isnan(source_emb.dot(target_emb)):
        print("not a number")
    score = sigmoid(math.pow(-1, label) * source_emb.dot(target_emb))
    g = math.pow(-1, label) * score * lr
    # total error for puser , luser and ruser vertex
    error += g * target_emb


# two stage method version 1
def update_group_item_twstage1(source, target, neg_vertices):
    error = np.zeros(DIM, )
    members = group_users.get(source)
    # group_embdding,member_index,exp(beta) matrix,member weight lamada
    source_emb, total_weight = cal_group_emb(members)
    target_emb = item_emb.get(target)
    update_item_vertex_in_group_stage1(source_emb, target_emb, error, 1)
    M = len(neg_vertices)
    if M != 0:
        for i in range(M):
            neg_emb = item_emb.get(neg_vertices[i])
            update_item_vertex_in_group_stage1(source_emb, neg_emb, error, 0)
    # error = sum((-1)^label*sigmoid((-1)^label*gm*vj)*vj) * lr
    new_user_weight_map = dict()
    new_feature_weight = dict()
    for feature in feature_weight.keys():
        new_feature_weight[feature] = 0
    for member in members:
        sum_weight = 0
        bias_weight = user_emb.get(member) / total_weight - source_emb / total_weight
        for feature, weight in feature_weight.items():
            sum_weight += user_feature_weight[str(member)].get(feature) * weight
        bias_regular = 2 * (user_weight_map.get(member) - sum_weight)
        # update user weight
        new_user_weight_map[member] = user_weight_map.get(member) - (error.dot(bias_weight) + reg * lr* bias_regular)
        for feature in feature_weight.keys():
            new_feature_weight[feature] += - 2 * reg * lr * (user_weight_map.get(member) - sum_weight)\
                                           * user_feature_weight[str(member)].get(feature)
        # update user embedding
    for feature,weight in new_feature_weight.items():
        feature_weight[feature] -= new_feature_weight[feature]
    for member in members:
        user_weight_map[member] = new_user_weight_map[member]


def update_group_item(source, target, neg_vertices):
    error = np.zeros(DIM, )
    members = group_users.get(source)
    # group_embdding,member_index,exp(beta) matrix,member weight lamada
    source_emb, total_weight = cal_group_emb(members)
    # print(source_emb)
    target_emb = item_emb.get(target)
    update_item_vertex_in_group(source_emb, target, target_emb, error, 1)
    M = len(neg_vertices)
    if M != 0:
        for i in range(M):
            neg_emb = item_emb.get(neg_vertices[i])
            update_item_vertex_in_group(source_emb, neg_vertices[i], neg_emb, error, 0)
    # error = sum((-1)^label*sigmoid((-1)^label*gm*vj)*vj) * lr
    # user weight lamda, user emb, feature weight
    new_puser_emb = dict()
    new_user_weight_map = dict()
    new_feature_weight = dict()
    for feature in feature_weight.keys():
        new_feature_weight[feature] = 0
    for member in members:
        sum_weight = 0
        bias_weight = user_emb.get(member) / total_weight - source_emb / total_weight
        for feature, weight in feature_weight.items():
            sum_weight += user_feature_weight[str(member)].get(feature) * weight
        bias_regular = 2 * (user_weight_map.get(member) - sum_weight)
        # update user weight
        new_user_weight_map[member] = user_weight_map.get(member) - (error.dot(bias_weight) + lr * reg * bias_regular)
        for feature in feature_weight.keys():
            new_feature_weight[feature] += - 2 * reg * lr * (user_weight_map.get(member) - sum_weight)\
                                           * user_feature_weight[str(member)].get(feature)
        # update user embedding
        new_puser_emb[member] = user_emb.get(member) - (user_weight_map.get(member) / total_weight) * error
    for feature,weight in new_feature_weight.items():
        feature_weight[feature] -= new_feature_weight[feature]
    for member in members:
        user_emb[member] = new_puser_emb[member]
        user_weight_map[member] = new_user_weight_map[member]


# fix source, sample targets
def neg_sample_user_item(source, target, source_nei):
    base_list = itematuser_list
    base_sample = pop_itematuser_sample
    # sample M negative vertices
    neg_vertices = list()
    record = 0
    while len(neg_vertices) < NEG_N:
        if record < len(base_list):
            sample_v = draw_vertex(base_list, base_sample)
            # sample_v = itematuser_neg_table[random.randint(0,neg_table_size-1)]
            if sample_v not in source_nei.get(source) and sample_v not in neg_vertices:
                neg_vertices.append(sample_v)
        else:
            break
        record += 1
    update_user_item(source, target, neg_vertices)


def neg_sample_group_item(source, target, source_nei, type):
    base_list = itematgroup_list
    #base_sample = pop_itematgroup_sample_dict.get(source)
    base_sample = pop_itematgroup_sample
    # sample M negative vertices
    neg_vertices = list()
    record = 0
    while len(neg_vertices) < NEG_N:
        if record < len(base_list):
            sample_v = draw_vertex(base_list, base_sample)
            # sample_v = itematgroup_neg_table[random.randint(0,neg_table_size-1)]
            if sample_v not in source_nei.get(source) and sample_v not in neg_vertices:
                neg_vertices.append(sample_v)
        else:
            break
        record += 1
    if type == 3 or type == 2:
        # update_group_item(source,target,neg_vertices)
        update_group_item(source, target, neg_vertices)
    else:
        update_group_item_twstage1(source, target, neg_vertices)


def training_user_item(tuple_list, user_nei):
    t = draw_tuple(tuple_list)
    v1 = t[0]
    v2 = t[1]
    # fix user, sample items
    neg_sample_user_item(v1, v2, user_nei)


def training_group_item(tuple_list, group_nei, type):
    while True:
        t = draw_tuple(tuple_list)
        # v1:group
        v1 = t[0]
        v2 = t[1]
        if len(group_users.get(v1)) < 50: break
    # fix group, sample items
    neg_sample_group_item(v1, v2, group_nei, type)


def bernoilli():
    r = random.random()
    if r < ita:
        return 1
    else:
        return 0


def train_data(type):
    if type == 1:
        # read trained user and event embedding
        get_emb(init_user,user_emb)
        get_emb(init_item,item_emb)
        print("The first stage training finished.")
        # train with group-event
        iter = 0
        last_count = 0
        current_sample_count = 0
        while iter <= group_event_N*10:
            if iter - last_count > 10000:
                current_sample_count += iter - last_count
                last_count = iter
                lr = init_lr * (1 - current_sample_count / (1.0 * (group_event_N + 1)))
                print("group event iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
                if lr < init_lr * 0.0001:
                    lr = init_lr * 0.0001
            if iter % (group_event_N) == 0 and iter != 0 and iter != group_event_N:
                # write embedding into file
                userweight_to_file(out_user_weight + str(iter), user_weight_map)
                featureweight_to_file(out_feature_weight + str(iter), feature_weight)
            training_group_item(all_edges_2, group_nei, type)
            print("group event iteration i:  %d finished." % iter)
            iter += 1
        # userweight_to_file(out_user_weight + str(group_event_N), user_weight_map)
        # featureweight_to_file(out_feature_weight + str(iter), feature_weight)
    elif type == 2:
        get_emb(init_user,user_emb)
        get_emb(init_item,item_emb)
        print("The first stage training finshed.")
        # train with group-event
        iter = 0
        last_count = 0
        current_sample_count = 0
        while iter <= group_event_N*10:
            if iter - last_count > 10000:
                current_sample_count += iter - last_count
                last_count = iter
                lr = init_lr * (1 - current_sample_count / (1.0 * (group_event_N + 1)))
                print("Iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
                if lr < init_lr * 0.0001:
                    lr = init_lr * 0.0001
            if iter % (group_event_N) == 0 and iter != 0:
                # write embedding into file
                emb_to_file(out_user + str(iter), user_emb)
                emb_to_file(out_item + str(iter), item_emb)
                userweight_to_file(out_user_weight + str(iter), user_weight_map)
                featureweight_to_file(out_feature_weight+str(iter),feature_weight)
            training_group_item(all_edges_2, group_nei, type)
            print("Iteration i:  %d finished." % iter)
            iter += 1
        # emb_to_file(out_user + str(group_event_N), user_emb)
        # emb_to_file(out_item + str(group_event_N), item_emb)
        # userweight_to_file(out_user_weight + str(group_event_N), user_weight_map)
        # featureweight_to_file(out_feature_weight + str(group_event_N), feature_weight)
    else:
        iter = 0
        last_count = 0
        current_sample_count = 0
        while iter <= (user_event_N * 30):
            if iter - last_count > 10000:
                current_sample_count += iter - last_count
                last_count = iter
                lr = init_lr * (1 - current_sample_count / (1.0 * (user_event_N + 1)))
                print("Iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
                if lr < init_lr * 0.0001:
                    lr = init_lr * 0.0001
            if iter % (user_event_N*2) == 0 and iter != 0:
                # write embedding into file
                emb_to_file(out_user + str(iter), user_emb)
                emb_to_file(out_item + str(iter), item_emb)
                userweight_to_file(out_user_weight + str(iter), user_weight_map)
                featureweight_to_file(out_feature_weight + str(iter), feature_weight)
            if bernoilli():
                training_group_item(all_edges_2, group_nei, type)
            else:
                training_user_item(all_edges_1, user_nei)
            if iter%10000 ==0:
                print("Iteration i:  %d finished." % iter)
            iter += 1


if __name__ == "__main__":
    get_pop_pro()
    #init_neg_table()
    print("initial neg table")
    init_all_vec()
    init_sigmod_table()
    print("training starting")
    train_data(2)
    print("training finished")
    emb_to_file(out_user+"_finished", user_emb)
    emb_to_file(out_item+"_finished", item_emb)
    userweight_to_file(out_user_weight+"_finished", user_weight_map)
    featureweight_to_file(out_feature_weight+"_finished",feature_weight)
