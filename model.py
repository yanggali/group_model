from file_process import read_file,get_group_users,write_to_file
import numpy as np
import random
import math
DIM = 64
N = 100000000
NEG_N = 2 #number of negative samples
init_lr = 0.025
lr = 0.0

#vertices and their degrees
user_degree = dict()
itematuser_degree = dict()
group_degree = dict()
itematgroup_degree = dict()

#all edges
all_edges_1 = list()
all_edges_2 = list()

#all vertices set
all_v = set()
all_user_v = set()
all_item_v = set()
all_luser_v = set()
all_ruser_v = set()

#vertex sample table
pop_user_sample = list()
pop_itematuser_sample = list()
pop_group_sample = list()
pop_itematgroup_sample = list()

#vertex and its neighbours
user_nei = dict()
itematuser_nei = dict()
group_nei = dict()
itematgroup_nei = dict()
group_users = dict()


#all vertices' embedding
all_v_emb = dict()
user_emb = dict()
item_emb = dict()
luser_emb = dict()
ruser_emb = dict()

SIGMOID_BOUND = 6
sigmoid_table_size = 1000
sig_map = dict()

#input files
inputdata1 = "data\\dataset\\kaggle\\train_user_event.dat"
inputdata2 = "data\\dataset\\kaggle\\train_groupid_event.dat"
inputdata3 = "data\\dataset\\kagle\\groupid_users.dat"
out_user = "data\\vectors\\kaggle\\user"
out_item = "data\\vectors\\kaggle\\item"
out_luser = "data\\vectors\\kaggle\\luser"
out_ruser = "data\\vectors\\kaggle\\ruser"

#初始化所有边，所有顶点的度数以及所有顶点的邻居

user_degree,itematuser_degree,user_nei,itematuser_nei,all_edges_1 = read_file(inputdata1,["userid","eventid"])
group_degree,itematgroup_degree,group_nei,itematgroup_nei,all_edges_2 = read_file(inputdata2,["groupid","eventid"])
group_users = get_group_users(inputdata3,["groupid","users"])
print("user set:%d ,item set:%d, group set:%d" %(len(user_degree.keys()),len(itematuser_degree.keys()),len(group_degree.keys())) )
user_list = list(user_degree.keys())
itematuser_list = list(itematuser_degree.keys())
group_list = list(group_degree.keys())
itematgroup_list = list(itematgroup_degree.keys())

def cal_pop_pro(vertex_list,vertex_degree,sample_list):
    totalweight = sum(vertex_degree.values())
    sample_list.append(float(vertex_degree[vertex_list[0]])/totalweight)
    for user in vertex_list[1:]:
        sample_list.append(float(pop_user_sample[-1] + vertex_degree[user])/totalweight)
    sample_list[-1] = 1.0

#initial sample table
def get_pop_pro():
    cal_pop_pro(user_list,user_degree,pop_user_sample)
    cal_pop_pro(itematuser_list,itematuser_degree,pop_itematuser_sample)
    cal_pop_pro(group_list, group_degree, pop_group_sample)
    cal_pop_pro(itematgroup_list, itematgroup_degree, pop_itematgroup_sample)

def gen_gaussian():
    max_value = 32767
    vector = np.zeros(DIM,)
    for i in range(DIM):
        vector[i] = (random.randint(0,max_value)*1.0/max_value-0.5)/DIM
    return vector

def init_vec(vec_list,vec_emb_dict):
    for vec in vec_list:
        vec_emb_dict[vec] = gen_gaussian()
#initial vertices' embedding
def init_all_vec():
    init_vec(user_list,user_emb)
    init_vec(itematuser_list,item_emb)
    useratgroup_set = set()
    for v in group_users.values():
        useratgroup_set.update(v)
    useratgroup_list = list(useratgroup_set)
    init_vec(useratgroup_list,luser_emb)
    init_vec(useratgroup_list, ruser_emb)


def init_sigmod_table():
    for k in range(sigmoid_table_size):
        x = 2*SIGMOID_BOUND*k/sigmoid_table_size-SIGMOID_BOUND
        sig_map[k] = 1/(1+math.exp(-x))


# sample an edge randomly
def draw_tuple(tuple_list):
    return tuple_list[random.randint(0,len(tuple_list))]


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
    k = int((x+SIGMOID_BOUND)*sigmoid_table_size/SIGMOID_BOUND/2)
    return sig_map.get(k)


# cal score according to two vertex,fix source ,sample target
def score_function(source,target,flag):
    if flag == 0:
        source_emb = user_emb.get(source)
        target_emb = item_emb.get(target)
    elif flag == 1:
        source_emb = item_emb.get(source)
        target_emb = user_emb.get(target)
    elif flag == 2:
        members = group_users.get(source)
        source_emb = cal_group_emb(members)
        target_emb = item_emb.get(target)
    else:
        source_emb = item_emb.get(source)
        members = group_users.get(target)
        target_emb = cal_group_emb(members)
    score = source_emb*target_emb
    return sigmoid(score)


def get_vec(v,flag):
    if flag == 0:
        return


# update user-item target vertex
def update_user_item_vertex(source,target,error,label,flag):
    if flag == 0:
        source_emb = user_emb.get(source)
        target_emb = item_emb.get(target)
        score = sigmoid(source_emb * target_emb)
        g = (label - score) * lr
        # total error for source vertex
        error += g * target_emb
        new_vec = target_emb + g * source_emb
        item_emb[target] = new_vec
    elif flag == 1:
        source_emb = item_emb.get(source)
        target_emb = user_emb.get(target)
        score = sigmoid(source_emb * target_emb)
        g = (label - score) * lr
        # total error for source vertex
        error += g * target_emb
        new_vec = target_emb + g * source_emb
        user_emb[target] = new_vec


def cal_group_emb(members):
    sum_weight = 0
    weight_dict = dict()
    for member1 in members:
        weight_dict[member1] = 0
        for member2 in members:
            if member1 != member2:
                weight = luser_emb.get(member1) * ruser_emb.get(member2)
                sum_weight += weight
                weight_dict[member1] += weight
    group_emb = np.zeros(DIM,)
    for member,weight in weight_dict.items():
        group_emb += weight*user_emb.get(member)
    return group_emb


def cal_group_totalweight(members):
    sum_weight = 0
    for member1 in members:
        for member2 in members:
            if member1 != member2:
                sum_weight += luser_emb.get(member1).dot(ruser_emb.get(member2))
    return sum_weight


def update_user_influ_emb(members,itemv_emb,g):
    for member1 in members:
        # update every member's luser and ruser embedding
        sum_weight = cal_group_totalweight(members)
        xr = np.zeros(DIM,)
        for other_member in members:
            if other_member != member1:
                xr += ruser_emb.get(other_member)
        g_luser_vector = np.zeros(DIM,)
        for member2 in members:
            if member1 == member2:
                g_luser_vector += (xr*sum_weight-luser_emb.get(member1).dot(xr)*xr).dot(user_emb.get(member1))
            else:
                x_r = np.zeros(DIM,)
                for member3 in member3:
                    if member3 != member2:
                        x_r += ruser_emb.get(member3)
                g_luser_vector += (0 - luser_emb.get(member2).dot(x_r)*xr).dot(user_emb.get(member2))
        g_luser_vector = (g_luser_vector / (sum_weight*sum_weight)).dot(itemv_emb)
        luser_emb[member1] = luser_emb.get(member1)+g*g_luser_vector

        # update ruser embedding
        sum_weight = cal_group_totalweight(members)
        xl = np.zeros(DIM,)
        for other_member in member3:
            if other_member != member1:
                xl += luser_emb.get(other_member)
        g_ruser_vector = np.zeros(DIM,)
        for member2 in members:
            self_emb = np.zeros(DIM, )
            for member in members:
                if member != member2:
                    self_emb += ruser_emb.get(member)
            if member2 == member1:
                self_emb = 0-luser_emb.get(member2).dot(self_emb)*xl.dot(user_emb.get(member2))
                g_ruser_vector += self_emb
            else:
                g_ruser_vector += (luser_emb.get(member2)*sum_weight - luser_emb.get(member2).dot(self_emb)*xl).dot(user_emb.get(member2))
        g_ruser_vector = (g_ruser_vector/(sum_weight*sum_weight)).dot(itemv_emb)
        ruser_emb[member1] = ruser_emb.get(member1) + g*g_ruser_vector


def update_group_item_vertex(source,target,error,label):
    source_emb = item_emb.get(source)
    members = group_users.get(target)
    target_emb = cal_group_emb(members)
    score = sigmoid(source_emb*target_emb)
    g = (label - score) * lr
    error += g * target_emb
    # update user influence embedding
    update_user_influ_emb(members,source_emb,g)


# update vertices according to an edge
def update_user_item(source,target,neg_vertices,flag):
    error = np.zeros(DIM,)
    update_user_item_vertex(source,target,error,1,flag)
    M = len(neg_vertices)
    if M != 0:
        for i in range(M):
            update_user_item_vertex(source,neg_vertices[i],error,0,flag)
    if flag == 0:
        new_vector = user_emb.get(source) + error
        user_emb[source] = new_vector
    else:
        new_vector = item_emb.get(source) + error
        item_emb[source] = new_vector

def update_item_vertex_in_group(source,target,luser_error,ruser_error,label):
    members = group_users.get(source)
    source_emb = cal_group_emb(members)
    target_emb = item_emb.get(target)
    score = sigmoid(source_emb * target_emb)
    g = (label - score) * lr
    new_vec = target_emb + g * source_emb
    item_emb[target] = new_vec
    # total error for luser and ruser vertex
    for member1 in members:
        # update every member's luser and ruser embedding
        sum_weight = cal_group_totalweight(members)
        xr = np.zeros(DIM,)
        for other_member in members:
            if other_member != member1:
                xr += ruser_emb.get(other_member)
        g_luser_vector = np.zeros(DIM,)
        for member2 in members:
            if member1 == member2:
                g_luser_vector += (xr*sum_weight-luser_emb.get(member1).dot(xr)*xr).dot(user_emb.get(member1))
            else:
                x_r = np.zeros(DIM,)
                for member3 in member3:
                    if member3 != member2:
                        x_r += ruser_emb.get(member3)
                g_luser_vector += (0 - luser_emb.get(member2).dot(x_r)*xr).dot(user_emb.get(member2))
        g_luser_error = (g_luser_vector / (sum_weight*sum_weight)).dot(target_emb)*g
        luser_error[member1] = g_luser_error

        xl = np.zeros(DIM, )
        for other_member in member3:
            if other_member != member1:
                xl += luser_emb.get(other_member)
        g_ruser_vector = np.zeros(DIM, )
        for member2 in members:
            self_emb = np.zeros(DIM, )
            for member in members:
                if member != member2:
                    self_emb += ruser_emb.get(member)
            if member2 == member1:
                self_emb = 0 - luser_emb.get(member2).dot(self_emb) * xl.dot(user_emb.get(member2))
                g_ruser_vector += self_emb
            else:
                g_ruser_vector += (luser_emb.get(member2) * sum_weight - luser_emb.get(member2).dot(self_emb) * xl).dot(
                    user_emb.get(member2))
        g_ruser_error = ((g_ruser_vector / (sum_weight * sum_weight)).dot(target_emb))*g
        ruser_error[member1] = g_ruser_error


def update_group_item(source,target,neg_vertices,flag):
    if flag == 0:
        luser_error_dict = dict()
        ruser_error_dict = dict()
        members = group_users.get(source)
        for m in members:
            luser_error_dict[m] = np.zeros(DIM,)
            ruser_error_dict[m] = np.zeros(DIM,)
            update_item_vertex_in_group(source, target, luser_error_dict,ruser_error_dict, 1)
        M = len(neg_vertices)
        if M != 0:
            for i in range(M):
                update_item_vertex_in_group(source, neg_vertices[i],luser_error_dict,ruser_error_dict, 0)
        for member in members:
            luser_emb[member] += luser_error_dict[member]
            ruser_emb[member] += ruser_error_dict[member]
    else:
        error = np.zeros(DIM,)
        update_group_item_vertex(source, target, error, 1)
        M = len(neg_vertices)
        if M != 0:
            for i in range(M):
                update_group_item_vertex(source, neg_vertices[i], error, 0)
        new_vector = item_emb.get(source) + error
        item_emb[source] = new_vector


# fix source, sample targets
def neg_sample_user_item(source,target,source_nei,flag):
    if flag == 0:
        base_list = itematuser_list
        base_sample = pop_itematuser_sample
    else:
        base_list = user_list
        base_sample = pop_user_sample
    # sample M negative vertices
    neg_vertices = list()
    record = 0
    while len(neg_vertices) < NEG_N:
        if record < len(base_list):
            sample_v = draw_vertex(base_list,base_sample)
            if sample_v not in source_nei.get(source) and sample_v not in neg_vertices:
                neg_vertices.append(sample_v)
        else: break
        record += 1
    update_user_item(source,target,neg_vertices,flag)


def neg_sample_group_item(source,target,source_nei,flag):
    if flag == 0:
        base_list = itematgroup_list
        base_sample = pop_itematgroup_sample
    else:
        base_list = group_list
        base_sample = pop_group_sample
    # sample M negative vertices
    neg_vertices = list()
    record = 0
    while len(neg_vertices) < NEG_N:
        if record < len(base_list):
            sample_v = draw_vertex(base_list,base_sample)
            if sample_v not in source_nei.get(source) and sample_v not in neg_vertices:
                neg_vertices.append(sample_v)
        else: break
        record += 1
    update_group_item(source,target,neg_vertices,flag)



def training_user_item(tuple_list,user_nei,item_nei):
    t = draw_tuple(tuple_list)
    v1 = t[0]
    v2 = t[1]
    # fix user, sample items
    neg_sample_user_item(v1,v2,user_nei,0)
    # fix item, sample users
    neg_sample_user_item(v2,v1,item_nei,1)


def training_group_item(tuple_list,group_nei,item_nei):
    t = draw_tuple(tuple_list)
    v1 = t[0]
    v2 = t[1]
    # fix group, sample items
    neg_sample_group_item(v1,v2,group_nei,0)

# training
def train_data():
    iter = 0
    last_count = 0
    current_sample_count = 0
    while iter <= N:
        if iter-last_count > 10000:
            current_sample_count += iter - last_count
            last_count = iter
            lr = init_lr * (1 - current_sample_count / (1.0 * (N + 1)))
            print("Iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
            if lr < init_lr*0.0001:
                lr = init_lr * 0.0001
        if iter%5000000 == 0 and iter != 0 and iter != N:
            # write embedding into file
            write_to_file(out_user+str(iter),user_emb)
            write_to_file(out_item+str(iter),item_emb)
            write_to_file(out_luser+str(iter),luser_emb)
            write_to_file(out_ruser+str(iter),ruser_emb)
        training_user_item(all_edges_1,user_nei,itematuser_nei)
        training_group_item(all_edges_2,group_nei,itematgroup_nei)


if __name__=="__main__":
    get_pop_pro()
    init_all_vec()
    init_sigmod_table()
    train_data()
    print("training finished")
    write_to_file(out_user,user_emb)
    write_to_file(out_item,item_emb)
    write_to_file(out_luser,luser_emb)
    write_to_file(out_ruser,ruser_emb)
