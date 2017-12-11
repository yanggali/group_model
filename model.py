from file_process import read_file,get_group_users,write_to_file
import numpy as np
import random
import math
DIM = 64
N = 100000000
NEG_N = 2 #number of negative samples
init_lr = 0.025
lr = init_lr
ita = 0.7
gama = 0
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

# vertex sample table
pop_user_sample = list()
pop_itematuser_sample = list()
pop_group_sample = list()
pop_itematgroup_sample = list()
# neg vertex sample table
itematuser_neg_table = list()
itematgroup_neg_table = list()
neg_table_size = 1e8
neg_sampling_power = 0.75

# vertex and its neighbours
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

#input sample files
inputdata1 = "data\\dataset\\kaggle\\sample_train_user_event.dat"
inputdata2 = "data\\dataset\\kaggle\\sample_train_groupid_event.dat"
inputdata3 = "data\\dataset\\kaggle\\sample_train_groupid_users.dat"
#input files
# inputdata1 = "data\\dataset\\kaggle\\train_user_event.dat"
# inputdata2 = "data\\dataset\\kaggle\\train_groupid_event.dat"
# inputdata3 = "data\\dataset\\kaggle\\groupid_users.dat"
out_user = "data\\vectors\\kaggle\\user"
out_item = "data\\vectors\\kaggle\\item"
out_luser = "data\\vectors\\kaggle\\luser"
out_ruser = "data\\vectors\\kaggle\\ruser"

#初始化所有边，所有顶点的度数以及所有顶点的邻居

user_degree,itematuser_degree,user_nei,itematuser_nei,all_edges_1 = read_file(inputdata1,["userid","eventid"])
itematgroup_degree,group_degree,itematgroup_nei,group_nei,all_edges_2 = read_file(inputdata2,["event","groupid"])
group_users = get_group_users(inputdata3,["groupid","users"])
print("user set:%d ,item set:%d, group set:%d" %(len(user_degree.keys()),len(itematuser_degree.keys()),len(group_degree.keys())) )
user_list = list(user_degree.keys())
itematuser_list = list(itematuser_degree.keys())
group_list = list(group_degree.keys())
itematgroup_list = list(itematgroup_degree.keys())


# 初始化所有负采样表
def init_vertex_neg_table(vertex_neg_table,vertex_degree,vertex_list)
    sum = 0.0,
    cur_num = 0.0
    por = 0.0
    vid = 0
    for vertex, degree in vertex_degree:
        sum += math.pow(degree, neg_sampling_power)
    for k in range(neg_table_size):
        if float(k + 1) / neg_table_size > por:
            cur_num += math.pow(vertex_degree.get(vertex_list[vid]), neg_sampling_power)
            por = cur_num / sum
            vid += 1
        vertex_neg_table[k] = vertex_list[vid - 1]


# 初始化负采样表
def init_neg_table():
    init_vertex_neg_table(itematuser_neg_table,itematuser_degree,itematuser_list)
    init_vertex_neg_table(itematgroup_neg_table, itematgroup_degree, itematgroup_list)



def cal_pop_pro(vertex_list,vertex_degree,sample_list):
    totalweight = sum(vertex_degree.values())
    sample_list.append(float(vertex_degree[vertex_list[0]])/totalweight)
    for vertex in vertex_list[1:]:
        sample_list.append(sample_list[-1] + float(vertex_degree[vertex])/totalweight)
    sample_list[-1] = 1.0


def cal_pop_pro_in_group(vertex_list,vertex_nei,group_sample_list_dict):
    for group,members in group_users.items():
        group_sample_list_dict[group] = list()
        for item in itematgroup_list:
            sum = 0.0
            for member in members:
                if member in vertex_nei[item]:
                    sum += 1

    totalweight = sum(vertex_degree.values())
    sample_list.append(float(vertex_degree[vertex_list[0]])/totalweight)
    for vertex in vertex_list[1:]:
        sample_list.append(sample_list[-1] + float(vertex_degree[vertex])/totalweight)
    sample_list[-1] = 1.0


#initial sample table
def get_pop_pro():
    cal_pop_pro(itematuser_list,itematuser_degree,pop_itematuser_sample)
    cal_pop_pro_in_group(itematgroup_list, itematgroup_degree, pop_itematgroup_sample)


def gen_gaussian():
    max_value = 32767
    vector = np.zeros(DIM,)
    for i in range(DIM):
        vector[i] = (random.randint(0,max_value)*1.0/max_value-0.5)/DIM
    return vector


def init_vec(vec_list,vec_emb_dict):
    for vec in vec_list:
        vec_emb_dict[vec] = gen_gaussian()


# initial vertices' embedding
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
    return tuple_list[random.randint(0,len(tuple_list)-1)]


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


# update user-item target vertex
def update_user_item_vertex(source,target,error,label):
    source_emb = user_emb.get(source)
    target_emb = item_emb.get(target)
    score = sigmoid(math.pow(-1,label)*source_emb.dot(target_emb))
    g = math.pow(-1,label) * score * lr
    # total error for source vertex
    error += g * target_emb
    new_vec = target_emb + g * source_emb
    item_emb[target] = new_vec


def cal_group_emb(members):
    sum_weight = 0
    weight_dict = dict()
    for member1 in members:
        weight_dict[member1] = 0
        for member2 in members:
            if member1 != member2:
                weight = luser_emb.get(member1).dot(ruser_emb.get(member2))
                sum_weight += weight
                weight_dict[member1] += weight
    group_emb = np.zeros(DIM,)
    for member,weight in weight_dict.items():
        group_emb += weight*user_emb.get(int(member))
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
                for member3 in members:
                    if member3 != member2:
                        x_r += ruser_emb.get(member3)
                g_luser_vector += (0 - luser_emb.get(member2).dot(x_r)*xr).dot(user_emb.get(member2))
        g_luser_vector = (g_luser_vector / (sum_weight*sum_weight)).dot(itemv_emb)
        luser_emb[member1] = luser_emb.get(member1)+g*g_luser_vector

        # update ruser embedding
        sum_weight = cal_group_totalweight(members)
        xl = np.zeros(DIM,)
        for other_member in members:
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
    print("group is %d" % target)
    members = group_users.get(target)
    target_emb = cal_group_emb(members)
    score = sigmoid(source_emb.dot(target_emb))
    g = (label - score) * lr
    error += g * target_emb
    # update user influence embedding
    update_user_influ_emb(members,source_emb,g)


# update vertices according to an edge
def update_user_item(source,target,neg_vertices):
    error = np.zeros(DIM,)
    update_user_item_vertex(source,target,error,1)
    M = len(neg_vertices)
    if M != 0:
        for i in range(M):
            update_user_item_vertex(source,neg_vertices[i],error,0)
    new_vector = user_emb.get(source) + error
    user_emb[source] = new_vector


def update_item_vertex_in_group(source,target,luser_error,ruser_error,label):
    members = group_users.get(source)
    source_emb = cal_group_emb(members)
    target_emb = item_emb.get(target)
    score = sigmoid(source_emb.dot(target_emb))
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
                for member3 in members:
                    if member3 != member2:
                        x_r += ruser_emb.get(member3)
                g_luser_vector += (0 - luser_emb.get(member2).dot(x_r)*xr).dot(user_emb.get(member2))
        g_luser_error = (g_luser_vector / (sum_weight*sum_weight)).dot(target_emb)*g
        luser_error[member1] = g_luser_error

        xl = np.zeros(DIM, )
        for other_member in members:
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
def neg_sample_user_item(source,target,source_nei):
    base_list = itematuser_list
    base_sample = pop_itematuser_sample
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
    update_user_item(source,target,neg_vertices)


def neg_sample_group_item(source,target,source_nei):
    base_list = itematgroup_list
    base_sample = pop_itematgroup_sample

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



def training_user_item(tuple_list,user_nei):
    t = draw_tuple(tuple_list)
    v1 = t[0]
    v2 = t[1]
    # fix user, sample items
    neg_sample_user_item(v1,v2,user_nei)


def training_group_item(tuple_list,group_nei,item_nei):
    t = draw_tuple(tuple_list)
    # v1:group
    v1 = t[1]
    v2 = t[0]
    # fix group, sample items
    neg_sample_group_item(v1,v2,group_nei)


def bernoilli():
    r = random.random()
    if r < ita: return 1
    else:return 0

# training
def train_data(type):
    if type == 0:
    elif type == 1:
    else:
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
            if bernoilli():
                training_group_item(all_edges_2, group_nei, itematgroup_nei)
            else:
                training_user_item(all_edges_1,user_nei)

            print("Iteration i:  %d finished." % iter)
            iter += 1


if __name__=="__main__":
    get_pop_pro()
    init_all_vec()
    init_sigmod_table()
    # init_neg_table()
    train_data(3)
    print("training finished")
    write_to_file(out_user,user_emb)
    write_to_file(out_item,item_emb)
    write_to_file(out_luser,luser_emb)
    write_to_file(out_ruser,ruser_emb)
