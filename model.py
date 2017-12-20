from file_process import read_file,get_group_users,write_to_file
import numpy as np
import random
import math
DIM = 50
NEG_N = 2 #number of negative samples
init_lr = 0.025
lr = init_lr
ita = 0.7
gama = 1
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
pop_itematuser_sample = list()
pop_itematgroup_sample_dict = dict()
pop_itematgroup_sample = list()
# neg vertex sample table
itematuser_neg_table = list()
itematgroup_neg_table = dict()
neg_table_size = int(1e8)
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

#input files
# inputdata1 = "data/dataset/kaggle/train_user_event.dat"
# inputdata2 = "data/dataset/kaggle/train_groupid_event.dat"
# inputdata3 = "data/dataset/kaggle/train_groupid_users.dat"
# out_user = "data/vectors/kaggle/user"
# out_item = "data/vectors/kaggle/item"
# out_luser = "data/vectors/kaggle/luser"
# out_ruser = "data/vectors/kaggle/ruser"


#input files
inputdata1 = "data/dataset/plancast/train_userid_eventid.dat"
inputdata2 = "data/dataset/plancast/train_groupid_eventid.dat"
inputdata3 = "data/dataset/plancast/train_groupid_userids.dat"
out_user = "data/vectors/plancast/user"
out_item = "data/vectors/plancast/item"
out_luser = "data/vectors/plancast/luser"
out_ruser = "data/vectors/plancast/ruser"

# initialize all edges, all vertices' degree and all vertices' neighbours
user_degree,itematuser_degree,user_nei,itematuser_nei,all_edges_1 = read_file(inputdata1,["userid","eventid"])
N = len(all_edges_1)*10
print("N is %d" % N)
print("user-event finished.")
group_degree,itematgroup_degree,group_nei,itematgroup_nei,all_edges_2 = read_file(inputdata2,["groupid","eventid"])
print("group-event finished.")
group_users = get_group_users(inputdata3,["groupid","users"])
print("user set:%d ,item set:%d, group set:%d" %(len(user_degree.keys()),len(itematuser_degree.keys()),len(group_degree.keys())) )
user_list = list(user_degree.keys())
itematuser_list = list(itematuser_degree.keys())
group_list = list(group_degree.keys())
itematgroup_list = list(itematgroup_degree.keys())
print("item size:%d" % len(itematuser_list))


# initial negative sampling table
def init_vertex_neg_table(vertex_neg_table,vertex_degree,vertex_list):
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
    # for user-event using negative sampling table
    init_vertex_neg_table(itematuser_neg_table,itematuser_degree,itematuser_list)
    # for group-event using common table
    cal_pop_pro_in_group(itematgroup_list, itematuser_nei, pop_itematgroup_sample_dict)


def cal_pop_pro(vertex_list,vertex_degree,sample_list):
    totalweight = 0.0
    for vertex,degree in vertex_degree.items():
        totalweight += math.pow(degree,neg_sampling_power)
    sample_list.append(float(math.pow(vertex_degree[vertex_list[0]],neg_sampling_power)/totalweight))
    for vertex in vertex_list[1:]:
        sample_list.append(sample_list[-1] + math.pow(float(vertex_degree[vertex]),neg_sampling_power)/totalweight)
    sample_list[-1] = 1.0


def cal_pop_pro_in_group(vertex_list,vertex_nei,group_sample_list_dict):
    for group,members in group_users.items():
        group_sample_list_dict[group] = list()
        for item in vertex_list:
            degree_sum = 0.0
            for member in members:
                if member in vertex_nei[item]:
                    degree_sum += 1
            group_sample_list_dict[group].append(math.pow(degree_sum+gama,neg_sampling_power))
    for group in group_sample_list_dict.keys():
        group_sample_list = group_sample_list_dict.get(group)
        totalweight = sum(group_sample_list)
        group_sample_list[0] /= totalweight
        for k in range(1,len(group_sample_list)):
            group_sample_list[k] = group_sample_list[k-1]+group_sample_list[k]/totalweight
        group_sample_list[-1] = 1.0
        group_sample_list_dict[group] = group_sample_list


# initial sample table
def get_pop_pro():
    cal_pop_pro(itematuser_list,itematuser_degree,pop_itematuser_sample)
    cal_pop_pro_in_group(itematgroup_list, itematuser_nei, pop_itematgroup_sample_dict)


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


# calculate groups' embedding
def cal_group_emb(members):
    sum_weight = 0
    index_member_dict = {members.index(member): member for member in members}
    member_index_dict = {member: members.index(member) for member in members}
    group_len = len(members)
    # calculate weight matrix
    weight_matrix = np.zeros((group_len, group_len))
    member_weight_dict = dict()
    for member1 in members:
        member_weight_dict[member1] = 0
        for member2 in members:
            if member1 != member2:
                weight = math.exp(luser_emb.get(member1).dot(ruser_emb.get(member2)))
                weight_matrix[member_index_dict[member1]][member_index_dict[member2]] = weight
                sum_weight += weight
                member_weight_dict[member1] += weight
    group_emb = np.zeros(DIM,)
    for member,weight in member_weight_dict.items():
        member_weight_dict[member] = weight / sum_weight
        group_emb += member_weight_dict[member]*user_emb.get(int(member))
    return group_emb,member_index_dict,weight_matrix,member_weight_dict


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


def update_item_vertex_in_group(source_emb,target,target_emb,error,label):
    score = sigmoid(math.pow(-1,label)*source_emb.dot(target_emb))
    g = math.pow(-1,label) * score * lr
    new_vec = target_emb + g * source_emb
    item_emb[target] = new_vec
    # total error for puser , luser and ruser vertex
    error += g * target_emb


# two stage method version 1
def update_group_item_twstage1(source,target,neg_vertices):
    error = np.zeros(DIM,)
    members = group_users.get(source)
    source_emb, member_index_dict, weight_matrix, member_weight_dict = cal_group_emb(members)
    target_emb = item_emb.get(target)
    score = sigmoid(math.pow(-1, 1) * source_emb.dot(target_emb))
    error = math.pow(-1, 1) * score * lr * target_emb
    M = len(neg_vertices)
    if M != 0:
        for i in range(M):
            neg_emb = item_emb.get(neg_vertices[i])
            error += math.pow(-1,0)*sigmoid(math.pow(-1, 0) * source_emb.dot(neg_emb))*lr*neg_emb
    new_luser_emb = dict()
    new_ruser_emb = dict()
    sum_weight = weight_matrix.sum()
    for member in members:
        member_index = member_index_dict.get(member)
        # update luser embedding
        l_l_part = np.zeros(DIM,)
        l_r_part = 0
        for member1 in members:
            if member1 != member:
                l_l_part += weight_matrix[member_index][member_index_dict[member1]]*ruser_emb.get(member1)
        for member_i_p in members:
            if member_i_p != member:
                for member_k in members:
                    if member_k != member_i_p:
                        l_r_part += weight_matrix[member_index_dict[member_i_p]][member_index_dict[member_k]]
        # luser_emb[member] = error.dot(user_emb.get(member))*l_part*r_part/(sum_weight*sum_weight)
        # update ruser embedding
        r_sum = 0
        for memberj in members:
            if memberj != member:
                r_l_part = np.zeros(DIM, )
                r_r_part = np.zeros(DIM, )
                sum = 0
                for member3 in members:
                    for member4 in members:
                        if member4 != member3:
                            sum += weight_matrix[member_index_dict[member3]][member_index_dict[member4]]
                r_l_part = weight_matrix[member_index_dict[memberj]][member_index_dict[member]]*luser_emb.get(memberj)*sum
                for member_i_p in members:
                    if member_i_p != member:
                        r_r_part += weight_matrix[member_index_dict[member_i_p]][member_index_dict[member]]*luser_emb.get(member_i_p)
                temp = 0
                for member_k in members:
                    if member_k != memberj:
                        temp += weight_matrix[member_index_dict[memberj]][member_index_dict[member_k]]
                r_r_part *= temp
                r_sum += user_emb.get(memberj).dot((r_l_part - r_r_part)/(sum_weight*sum_weight))
        new_ruser_emb[member] = ruser_emb.get(member)+error * r_sum
        new_luser_emb[member] = luser_emb.get(member) + error.dot(user_emb.get(member)) * l_l_part * l_r_part / (sum_weight * sum_weight)
        # update user embedding
    for member in members:
        luser_emb[member] = new_luser_emb[member]
        ruser_emb[member] = new_ruser_emb[member]


def update_group_item(source,target,neg_vertices):
    error = np.zeros(DIM,)
    members = group_users.get(source)
    # group_embdding,member_index,exp(beta)矩阵,member权重lamada
    source_emb,member_index_dict,weight_matrix,member_weight_dict = cal_group_emb(members)
    target_emb = item_emb.get(target)
    update_item_vertex_in_group(source_emb, target,target_emb, error, 1)
    M = len(neg_vertices)
    if M != 0:
        for i in range(M):
            neg_emb = item_emb.get(neg_vertices[i])
            update_item_vertex_in_group(source_emb,neg_vertices[i],neg_emb, error, 0)
    # error = sum((-1)^label*sigmoid((-1)^label*gm*vj)*vj) * lr
    sum_weight = weight_matrix.sum()
    new_ruser_emb = dict()
    new_luser_emb = dict()
    new_puser_emb = dict()
    for member in members:
        member_index = member_index_dict.get(member)
        # update luser embedding
        l_l_part = np.zeros(DIM,)
        l_r_part = 0
        for member1 in members:
            if member1 != member:
                l_l_part += weight_matrix[member_index][member_index_dict[member1]]*ruser_emb.get(member1)
        for member_i_p in members:
            if member_i_p != member:
                for member_k in members:
                    if member_k != member_i_p:
                        l_r_part += weight_matrix[member_index_dict[member_i_p]][member_index_dict[member_k]]
        # luser_emb[member] = error.dot(user_emb.get(member))*l_part*r_part/(sum_weight*sum_weight)
        # update ruser embedding
        r_sum = 0
        for memberj in members:
            if memberj != member:
                r_l_part = np.zeros(DIM, )
                r_r_part = np.zeros(DIM, )
                sum = 0
                for member3 in members:
                    for member4 in members:
                        if member4 != member3:
                            sum += weight_matrix[member_index_dict[member3]][member_index_dict[member4]]
                r_l_part = weight_matrix[member_index_dict[memberj]][member_index_dict[member]]*luser_emb.get(memberj)*sum
                for member_i_p in members:
                    if member_i_p != member:
                        r_r_part += weight_matrix[member_index_dict[member_i_p]][member_index_dict[member]]*luser_emb.get(member_i_p)
                temp = 0
                for member_k in members:
                    if member_k != memberj:
                        temp += weight_matrix[member_index_dict[memberj]][member_index_dict[member_k]]
                r_r_part *= temp
                r_sum += user_emb.get(memberj).dot((r_l_part - r_r_part)/(sum_weight*sum_weight))
        new_ruser_emb[member] = ruser_emb.get(member)+error * r_sum
        new_luser_emb[member] = luser_emb.get(member) + error.dot(user_emb.get(member)) * l_l_part * l_r_part / (sum_weight * sum_weight)
        # update user embedding
        new_puser_emb[member] = user_emb.get(member) + member_weight_dict.get(member) * error
    for member in members:
        luser_emb[member] = new_luser_emb[member]
        ruser_emb[member] = new_ruser_emb[member]
        user_emb[member] = new_puser_emb[member]


# fix source, sample targets
def neg_sample_user_item(source,target,source_nei):
    base_list = itematuser_list
    # base_sample = pop_itematuser_sample
    # sample M negative vertices
    neg_vertices = list()
    record = 0
    while len(neg_vertices) < NEG_N:
        if record < len(base_list):
            # sample_v = draw_vertex(base_list,base_sample)
            sample_v = itematuser_neg_table[random.randint(0,neg_table_size-1)]
            if sample_v not in source_nei.get(source) and sample_v not in neg_vertices:
                neg_vertices.append(sample_v)
        else: break
        record += 1
    update_user_item(source,target,neg_vertices)


def neg_sample_group_item(source,target,source_nei,type):
    base_list = itematgroup_list
    base_sample = pop_itematgroup_sample_dict.get(source)
    # sample M negative vertices
    neg_vertices = list()
    record = 0
    while len(neg_vertices) < NEG_N:
        if record < len(base_list):
            sample_v = draw_vertex(base_list,base_sample)
            # sample_v = itematgroup_neg_table[random.randint(0,neg_table_size-1)]
            if sample_v not in source_nei.get(source) and sample_v not in neg_vertices:
                neg_vertices.append(sample_v)
        else: break
        record += 1
    if type == 3 or type == 2:
        update_group_item(source,target,neg_vertices)
    else:
        update_group_item_twstage1(source,target,neg_vertices)


def training_user_item(tuple_list,user_nei):
    t = draw_tuple(tuple_list)
    v1 = t[0]
    v2 = t[1]
    # fix user, sample items
    neg_sample_user_item(v1,v2,user_nei)


def training_group_item(tuple_list,group_nei,type):
    while True:
        t = draw_tuple(tuple_list)
        # v1:group
        v1 = t[0]
        v2 = t[1]
        if len(group_users.get(v1)) < 50:break
    # fix group, sample items
    neg_sample_group_item(v1,v2,group_nei,type)

def bernoilli():
    r = random.random()
    if r < ita: return 1
    else:return 0


# training
def train_data(type):
    if type == 1:
        iter = 0
        last_count = 0
        current_sample_count = 0
        while iter <= N:
            if iter - last_count > 10000:
                current_sample_count += iter - last_count
                last_count = iter
                lr = init_lr * (1 - current_sample_count / (1.0 * (N + 1)))
                print("Iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
                if lr < init_lr * 0.0001:
                    lr = init_lr * 0.0001
            if iter % (N/10.0) == 0 and iter != 0 and iter != N:
                # write embedding into file
                write_to_file(out_user + str(iter), user_emb)
                write_to_file(out_item + str(iter), item_emb)
            training_user_item(all_edges_1, user_nei)
            print("Iteration i:  %d finished." % iter)
            iter += 1
        print("The first stage training finished.")
        # train with group-event
        iter = 0
        last_count = 0
        current_sample_count = 0
        while iter <= N:
            if iter - last_count > 10000:
                current_sample_count += iter - last_count
                last_count = iter
                lr = init_lr * (1 - current_sample_count / (1.0 * (N + 1)))
                print("Iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
                if lr < init_lr * 0.0001:
                    lr = init_lr * 0.0001
            if iter % (N/10) == 0 and iter != 0 and iter != N:
                # write embedding into file
                write_to_file(out_luser + str(iter), luser_emb)
                write_to_file(out_ruser + str(iter), ruser_emb)
            training_group_item(all_edges_2, group_nei,type)
            print("Iteration i:  %d finished." % iter)
            iter += 1
    elif type == 2:
        iter = 0
        last_count = 0
        current_sample_count = 0
        while iter <= N:
            if iter - last_count > 10000:
                current_sample_count += iter - last_count
                last_count = iter
                lr = init_lr * (1 - current_sample_count / (1.0 * (N + 1)))
                print("Iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
                if lr < init_lr * 0.0001:
                    lr = init_lr * 0.0001
            # if iter % 5000000 == 0 and iter != 0 and iter != N:
            #     # write embedding into file
            #     write_to_file(out_user + str(iter), user_emb)
            #     write_to_file(out_item + str(iter), item_emb)
            training_user_item(all_edges_1, user_nei)
            print("Iteration i:  %d finished." % iter)
            iter += 1
        print("The first stage training finshed.")
        # train with group-event
        iter = 0
        last_count = 0
        current_sample_count = 0
        while iter <= N:
            if iter - last_count > 10000:
                current_sample_count += iter - last_count
                last_count = iter
                lr = init_lr * (1 - current_sample_count / (1.0 * (N + 1)))
                print("Iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
                if lr < init_lr * 0.0001:
                    lr = init_lr * 0.0001
            if iter % (N/10) == 0 and iter != 0 and iter != N:
                # write embedding into file
                write_to_file(out_user + str(iter), user_emb)
                write_to_file(out_item + str(iter), item_emb)
                write_to_file(out_luser + str(iter), luser_emb)
                write_to_file(out_ruser + str(iter), ruser_emb)
            training_group_item(all_edges_2, group_nei, type)
            print("Iteration i:  %d finished." % iter)
            iter += 1
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
            if iter%(int(N/100)) == 0 and iter != 0 and iter != N:
                # write embedding into file
                write_to_file(out_user+str(iter),user_emb)
                write_to_file(out_item+str(iter),item_emb)
                write_to_file(out_luser+str(iter),luser_emb)
                write_to_file(out_ruser+str(iter),ruser_emb)
            if bernoilli():
                training_group_item(all_edges_2, group_nei,type)
            else:
                training_user_item(all_edges_1,user_nei)

            print("Iteration i:  %d finished." % iter)
            iter += 1


if __name__=="__main__":
    # get_pop_pro()
    init_neg_table()
    print("initial neg table")
    init_all_vec()
    init_sigmod_table()
    print("training starting")
    train_data(3)
    print("training finished")
    write_to_file(out_user,user_emb)
    write_to_file(out_item,item_emb)
    write_to_file(out_luser,luser_emb)
    write_to_file(out_ruser,ruser_emb)
