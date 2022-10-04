# -*- coding:  UTF-8 -*-
from __future__ import division
import math
from metric import *
# from ops import *
import time
import numpy as np

from metric import *
import tensorflow as tf
import pandas as pd
import tensorflow.keras.backend as KK
import os

# main function
from Graph_Model2 import QueryModel
# from Graph_Model2 import QueryModel2
from Graph_Model2 import Decoder
from relation_POI import load_data
import data_augmentation1
from Graph_Model2 import EMModel
from Graph_Model2 import MLP
from Graph_Model2 import PreMlp
from Graph_Model2 import Category_Decoder

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Edit by Qiang Gao,2021,Jul,3
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# ---------------------------------------
tf.keras.backend.set_floatx('float64')
# Hyper-parameters
dynamic_traning = True
batch_size = 16
pre_batch_size = 16
# ----------------------------------------
# =============================== datasets ====================================== #
dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb', 'TKY_split200']
dat_ix = 0
poi_name = "poi-"+dat_suffix[dat_ix]+".csv"
tra_name = "traj-"+dat_suffix[dat_ix]+".csv"
embedding_name = dat_suffix[dat_ix]
model = './logs/model_'+embedding_name+'.pkt'

# =============================== data load ====================================== #
#load original data
op_tdata = open('origin_data/'+poi_name, 'r')
ot_tdata = open('origin_data/'+tra_name, 'r')
print ('To Train',dat_suffix[dat_ix])
POIs=[]
Trajectory=[]
IDs=[]
for line in op_tdata.readlines():
    lineArr = line.split(',')
    temp_line=list()
    for item in lineArr:
        temp_line.append(item.strip('\n'))
    POIs.append(temp_line)
POIs=POIs[1:] #remove first line

#calculate the distance between two places
def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088 # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist = 2 * radius * np.arcsin( np.sqrt(
                (np.sin(0.5*dlat))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5*dlng))**2 ))
    return dist

# =============================== POI ====================================== #
get_POIs={}
char_pois=[] #pois chars

for items in POIs:
    char_pois.append(items[0])
    get_POIs.setdefault(items[0],[]).append([items[2],items[3]]) # pois to category
Users=[]
poi_count={}
for line in ot_tdata.readlines():
    lineArr=line.split(',')
    temp_line=list()
    if lineArr[0]=='userID':
        continue
    poi_count.setdefault(lineArr[2], []).append(lineArr[2])
    IDs.append(lineArr[1])
    for i in range(len(lineArr)):
        if i==0:
            user = lineArr[i]
            Users.append(user)  # add user id
            temp_line.append(user)
            continue
        temp_line.append(lineArr[i].strip('\n'))
    Trajectory.append(temp_line)
Users=sorted(list(set(Users)))
print ('user number',len(Users))

TRAIN_TRA=[]
TRAIN_USER=[]
TRAIN_TIME=[]
TRAIN_DIST=[]
DATA={} #temp_data
print ('original data numbers',len(Trajectory))

#remove the trajectory which has less than three POIs
for index in range(len(Trajectory)):
    if(int(Trajectory[index][-2])>=3): #the length of the trajectory must over than 3
        DATA.setdefault(Trajectory[index][0]+'-'+Trajectory[index][1],[]).append([Trajectory[index][2],Trajectory[index][3],Trajectory[index][4]]) #userID+trajID

#print (len(IDs))
IDs=set(IDs)
print ('original trajectory numbers',len(IDs)) #trajectory id

distance_count=[]
for key in DATA.keys():
    traj=DATA[key]
    #print traj
    for i in range(len(traj)):
        #print get_POIs[traj[i][0]][0][0]
        lon1=float(get_POIs[traj[i][0]][0][0])
        lat1=float(get_POIs[traj[i][0]][0][1])
        for j in range(i+1,len(traj)):
            lon2 = float(get_POIs[traj[j][0]][0][0])
            lat2 = float(get_POIs[traj[j][0]][0][1])
            distance_count.append(calc_dist_vec(lon1,lat1,lon2,lat2))
upper_dis=max(distance_count)
lower_dis=min(distance_count)
print ('distance between two POIs',len(distance_count))

for keys in DATA.keys():
    user_traj=DATA[keys]
    temp_poi=[]
    temp_time=[]
    temp_dist=[]
    for i in range(len(user_traj)):
        temp_poi.append(user_traj[i][0]) #add poi id
        lon1=float(get_POIs[user_traj[i][0]][0][0])
        lat1=float(get_POIs[user_traj[i][0]][0][1])

        #start point
        lons=float(get_POIs[user_traj[0][0]][0][0])
        lats=float(get_POIs[user_traj[0][0]][0][1])

        #end point
        lone=float(get_POIs[user_traj[-1][0]][0][0])
        late=float(get_POIs[user_traj[-1][0]][0][1])

        sd=calc_dist_vec(lon1,lat1,lons,lats)
        ed = calc_dist_vec(lon1, lat1, lone, late)
        value1=0.5*(sd)/max(distance_count)
        value2=0.5*(ed)/max(distance_count)
        #print value
        temp_dist.append([value1,value2]) #lon,lat

        dt = time.strftime("%H:%M:%S", time.localtime(int(user_traj[i][1:][0])))
        #print dt.split(":")[0]
        temp_time.append(int(dt.split(":")[0])) #add poi time
    TRAIN_USER.append(keys)
    TRAIN_TRA.append(temp_poi)
    TRAIN_TIME.append(temp_time)
    TRAIN_DIST.append(temp_dist)
dictionary={}
for key in poi_count.keys():
    count=len(poi_count[key])
    dictionary[key]=count
#dictionary['GO']=1
#dictionary['PAD']=1
dictionary['END']=1
new_dict=sorted(dictionary.items(),key = lambda x:x[1],reverse = True)

print('poi number is',len(new_dict)-1) # three vitual POIs
voc_poi = list()

for item in new_dict:
    voc_poi.append(item[0]) #has been sorted by frequency

#extract real POI id to a fake id
def extract_words_vocab():
    int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

int_to_vocab, vocab_to_int = extract_words_vocab()

#generate pre-traning dataset withGO,END,PAD
new_trainT = list()
for i in range(len(TRAIN_TRA)): #TRAIN
    temp = list()
    #temp.append(vocab_to_int['GO'])
    for j in range(len(TRAIN_TRA[i])):
        temp.append(vocab_to_int[TRAIN_TRA[i][j]])
    temp.append(vocab_to_int['END'])
    #temp.append(vocab_to_int['PAD'])
    new_trainT.append(temp)

# print (new_trainT[0]) #e.g. GO,1,2,3,END,PAD
# print (new_trainT[1])

#generate traning dataset without GO,END,PAD
new_trainTs = list()
for i in range(len(TRAIN_TRA)): #TRAIN
    temp = list()
    for j in range(len(TRAIN_TRA[i])):
        temp.append(vocab_to_int[TRAIN_TRA[i][j]])
    new_trainTs.append(temp)


#output the trajectory
dataset = open('data/'+embedding_name+'_set.dat', 'w')
for i in range(len(new_trainTs)):
    dataset.write(str(TRAIN_USER[i])+'\t')
    for j in range(len(new_trainTs[i])):
        dataset.write(str(new_trainTs[i][j])+'\t')
    dataset.write('\n')
dataset.close()


#basic function
def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])  # 取最大长度
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def pad_time_batch(time_batch):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in time_batch])  # 取最大长度
    return [sentence + [0] * (max_sentence - len(sentence)) for sentence in time_batch]


def pad_dist_batch(dist_batch):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in dist_batch])  # 取最大长度
    return [sentence + [sentence[-1]] * (max_sentence - len(sentence)) for sentence in dist_batch]


def eos_sentence_batch(sentence_batch, eos_in):
    return [sentence + [eos_in] for sentence in sentence_batch]


# main function
def get_data(index, K):
    # sort original data
    index_T = {}
    trainT = []
    trainU = []
    trainTime=[]
    trainDist=[]
    for i in range(len(new_trainTs)):
        index_T[i] = len(new_trainTs[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        id = temp_size[i][0]
        trainT.append(new_trainTs[id])
        trainU.append(TRAIN_USER[id])
        trainTime.append(TRAIN_TIME[id])
        trainDist.append(TRAIN_DIST[id])
    value=int(math.ceil(len(trainT)/K))
    if index==K-1:
        testT=trainT[-value:]
        testU=trainU[-value:]
        trainT=trainT[:-value]
        trainU=trainU[:-value]

        testTime=trainTime[-value:]
        testDist=trainDist[-value:]
        trainTime=trainTime[:-value]
        trainDist=trainDist[:-value]

    elif index==0:
        testT=trainT[:(index+1)*value]
        testU=trainU[:(index+1)*value]
        trainT =trainT[(index+1)*value:]
        trainU =trainU[(index+1)*value:]

        testTime=trainTime[:(index+1)*value]
        testDist=trainDist[:(index+1)*value]
        trainTime=trainTime[(index+1)*value:]
        trainDist=trainDist[(index+1)*value:]

    else:
        testT=trainT[index*value:(index+1)*value]
        testU=trainU[index*value:(index+1)*value]
        trainT = trainT[0:index*value]+trainT[(index+1)*value:]
        trainU = trainU[0:index*value]+trainU[(index+1)*value:]

        testTime=trainTime[index*value:(index+1)*value]
        testDist=trainDist[index*value:(index+1)*value]
        trainTime=trainTime[0:index*value]+trainTime[(index+1)*value:]
        trainDist=trainDist[0:index*value]+trainDist[(index+1)*value:]
    train_size = len(trainT) % batch_size
    #if
    trainT = trainT + [trainT[-1]]*(batch_size-train_size)  # copy data and fill the last batch size
    trainU = trainU + [trainU[-1]]*(batch_size-train_size)
    trainTime=trainTime+[trainTime[-1]]*(batch_size-train_size)
    trainDist = trainDist + [trainDist[-1]] * (batch_size - train_size)
    #print 'Text', testT,index,K
    test_size = len(testT) % batch_size
    if test_size!=0:
        testT = testT + [testT[-1]]*(batch_size-test_size)  # copy data and fill the last batch size
        testU = testU + [testU[-1]]*(batch_size-test_size)  #BUG for test_size<batch_size len(train_size<test_size)
        testTime=testTime+[testTime[-1]]*(batch_size-test_size)
        testDist = testDist + [testDist[-1]] * (batch_size - test_size)
    print ('test size',test_size,len(testT))
    #pre-processing
    step=0
    encoder_train=[]
    decoder_trian=[]
    encoder_test=[]
    decoder_test=[]
    n_trainTime=[]
    n_testTime=[]
    n_trainDist1=[]
    n_trainDist2= []
    n_testDist1=[]
    n_testDist2= []
    train_batch_lenth=[]
    test_batch_lenth=[]
    z_train=[]
    z_train_time=[]
    z_train_dist1=[]
    z_train_dist2 = []
    z_test=[]
    z_test_time=[]
    z_test_dist1=[]
    z_test_dist2 =[]
    while step < len(trainU) // batch_size:
        start_i = step * batch_size
        input_x = trainT[start_i:start_i + batch_size]
        #time
        input_time = trainTime[start_i:start_i + batch_size]
        input_time_ = pad_time_batch(input_time)
        input_d = trainDist[start_i:start_i + batch_size]
        #input
        encode_batch = pad_sentence_batch(input_x, vocab_to_int['END'])
        decode_batchs = []
        z_batch=[]
        z_batch_time=[]
        z_batch_dist1=[]
        z_batch_dist2=[]
        for sampe in input_x:
            value = sampe
            value_=[sampe[0],sampe[-1]]
            decode_batchs.append(value)
            z_batch.append(value_)
        for sample in input_time:
            z_batch_time.append([sample[0],sample[-1]])
        decode_batch_ = eos_sentence_batch(decode_batchs, vocab_to_int['END'])
        decode_batch = pad_sentence_batch(decode_batch_, vocab_to_int['END'])

        dist_1 = []
        dist_2 = []
        # print 'value',input_d
        for i in range(len(input_d)):
            temp_dist1 = []
            temp_dist2 = []
            for j in range(len(input_d[i])):
                temp_dist1.append(input_d[i][j][0])
                temp_dist2.append(input_d[i][j][1])
            dist_1.append(temp_dist1)
            dist_2.append(temp_dist2)
            z_batch_dist1.append([temp_dist1[0],temp_dist1[-1]])
            z_batch_dist2.append([temp_dist2[0], temp_dist2[-1]])
        dist_1_ = pad_dist_batch(dist_1)
        dist_2_ = pad_dist_batch((dist_2))

        pad_source_lengths = []
        for source in decode_batchs:
            pad_source_lengths.append(len(source) + 1)
        for i in range(batch_size):
            encoder_train.append(encode_batch[i])
            decoder_trian.append(decode_batch[i])
            train_batch_lenth.append(pad_source_lengths[i])
            n_trainTime.append(input_time_[i])
            n_trainDist1.append(dist_1_[i])
            n_trainDist2.append(dist_2_[i])
            z_train.append(z_batch[i])
            z_train_time.append(z_batch_time[i])
            z_train_dist1.append(z_batch_dist1[i])
            z_train_dist2.append(z_batch_dist2[i])
        step+=1
        #append to
    steps=0
    while steps < len(testU) // batch_size:
        start_i = steps * batch_size
        input_x = testT[start_i:start_i + batch_size]
        # time
        input_time = testTime[start_i:start_i + batch_size]
        input_time_ = pad_time_batch(input_time)
        input_d = testDist[start_i:start_i + batch_size]
        # input
        encode_batch = pad_sentence_batch(input_x, vocab_to_int['END'])
        decode_batchs = []
        z_batch = []
        z_batch_time = []
        z_batch_dist1 = []
        z_batch_dist2 = []
        for sampe in input_x:
            value = sampe
            value_ = [sampe[0], sampe[-1]]
            decode_batchs.append(value)
            z_batch.append(value_)
        for sample in input_time:
            z_batch_time.append([sample[0], sample[-1]])
        decode_batch_ = eos_sentence_batch(decode_batchs, vocab_to_int['END'])
        decode_batch = pad_sentence_batch(decode_batch_, vocab_to_int['END'])

        dist_1 = []
        dist_2 = []
        # print 'value',input_d
        for i in range(len(input_d)):
            temp_dist1 = []
            temp_dist2 = []
            for j in range(len(input_d[i])):
                temp_dist1.append(input_d[i][j][0])
                temp_dist2.append(input_d[i][j][1])
            dist_1.append(temp_dist1)
            dist_2.append(temp_dist2)
            z_batch_dist1.append([temp_dist1[0], temp_dist1[-1]])
            z_batch_dist2.append([temp_dist2[0], temp_dist2[-1]])
        dist_1_ = pad_dist_batch(dist_1)
        dist_2_ = pad_dist_batch((dist_2))

        pad_source_lengths = []
        for source in decode_batchs:
            pad_source_lengths.append(len(source) + 1)
        for i in range(batch_size):
            encoder_test.append(encode_batch[i])
            decoder_test.append(decode_batch[i])
            test_batch_lenth.append(pad_source_lengths[i])
            n_testTime.append(input_time_[i])
            n_testDist1.append(dist_1_)
            n_testDist2.append(dist_2_)
            z_test.append(z_batch[i])
            z_test_time.append(z_batch_time[i])
            z_test_dist1.append(z_batch_dist1[i])
            z_test_dist2.append(z_batch_dist2[i])
        steps+=1
    train_variables=[encoder_train,decoder_trian,train_batch_lenth, n_trainTime,n_trainDist1, n_trainDist2,z_train,z_train_time,z_train_dist1,z_train_dist2]
    test_variables = [encoder_test, decoder_test, test_batch_lenth, n_testTime, n_testDist1, n_testDist2,
                       z_test, z_test_time, z_test_dist1, z_test_dist2]
    return train_variables,test_variables


# main--

# optimizer
learning_rate = 0.1
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, vocab_to_int['END'])) # id end is 29
    # print (mask)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    # mask the pad
    loss_ *= mask
    return tf.reduce_mean(loss_)


def pre_loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, vocab_to_int['END']))
    cat_real = []
    for poiid in real.numpy():
        if(poiid == vocab_to_int['END']):
            cat = 0
        else:
            cat = poiint_to_catid[poiid]
        cat_real.append(cat)
    cat_real = tf.cast(cat_real, dtype='int32')
    # print (mask)
    loss_ = loss_object(cat_real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    # mask the pad
    loss_ *= mask
    return tf.reduce_mean(loss_)

def cat_loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, vocab_to_int['END']))
    cat_real = []
    for poiid in real:
        if(poiid == vocab_to_int['END']):
            cat = 0
        else:
            cat = poiint_to_catid[poiid]
        cat_real.append(cat)
    cat_real = tf.cast(cat_real, dtype='int32')
    # print (mask)
    loss_ = loss_object(cat_real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    # mask the pad
    loss_ *= mask
    return tf.reduce_mean(loss_)


# =============================== Graph Embedding Layer ====================================== #
A_data = load_data(dat_suffix[dat_ix])
A_hat = A_data.load_A()
poi_time = A_data.load_poi_time()

q_dim = 256
dec_units = 256
cat_dec_units = 32
POI_embedding_size = 256
time_embedding_size = 256
cat_embedding_size = 32
heuristic_prior_weight = 0.1
# =============================== POI Embedding Layer ====================================== #
# embeddings
# print (len(voc_poi)-2)
# if dynamic_traning is True:
#     location_embeddings=np.random.uniform(-1,1,size=(len(voc_poi),POI_embedding_size))
#     time_embeddings=np.random.uniform(-1,1,size=(24,32))
#     location_embeddings,time_embeddings=EMModel(len(voc_poi)-2,24,POI_embedding_size,32)

poi_size = len(voc_poi)
# len(location_embeddings) #there is a fake location that is PAD
print('poi_size', poi_size)
# embedding_layer=EMModel(poi_size,24,POI_embedding_size,time_embedding_size)


def next_poi_probability():
    '''
    统计poi点出现的个数，KBS work 启发式
    生成与evaluate和train的prediction相同维度
    '''
    city = dat_suffix[dat_ix]
    poi_probability = np.zeros(poi_size)
    file = open('./traj_data/' + city + '-trajs.dat', 'r')
    for line in file.readlines():
        traj = line.split()
        traj = [vocab_to_int[i] for i in traj]
        traj = traj[1:-1]
        for poi in traj:
            poi_probability[poi] += 1

    return poi_probability / poi_probability.sum()


poi_probability = next_poi_probability()
heuristic_prior = tf.cast(poi_probability, dtype='float64')
print('heuristic_prior', heuristic_prior)

def poi_category_dict():
    '''
    获得poi_number对应的种类
    :return: 字典，种类个数
    '''
    city = dat_suffix[dat_ix]
    data = pd.read_csv('./origin_data/poi-'+city+'.csv')
    category_list = set(data['poiCat'])
    category_number = len(category_list)

    category_id = [i for i in range(category_number)]
    category_to_categoryid = dict(zip(category_list, category_id))
    poiid_to_category = dict(zip(data['poiID'], data['poiCat']))
    poiint_to_categoryid = {}
    for poiid in vocab_to_int.keys():
        if(poiid == 'END'):
            continue
        poiint = vocab_to_int[poiid]
        categoryid = category_to_categoryid[poiid_to_category[eval(poiid)]]
        poiint_to_categoryid[poiint] = categoryid

    return poiint_to_categoryid, category_number


poiint_to_catid, cat_size = poi_category_dict()
print(poiint_to_catid)
embedding_layer = EMModel(poi_size, 24, POI_embedding_size, time_embedding_size,
                          cat_size, cat_embedding_size, A_hat, poi_time)
query = QueryModel(POI_embedding_size, time_embedding_size, q_dim)
decoder = Decoder(poi_size, dec_units)
cat_decoder = Category_Decoder(cat_size, poiint_to_catid, cat_dec_units)
mlp = MLP(poi_size)
pre_mlp = PreMlp()


# --------------------------- train_step ---------------------------------
def train_step(que, traj, pad_lengths):
    loss = 0
    cat_loss = 0
    total_loss = 0
    with tf.GradientTape() as tape:
        # tf.print(embedding_layer())
        location_embeddings, time_embeddings, cat_embedding = embedding_layer(dynamic_traning)
        query_out = query(que, location_embeddings, time_embeddings)
        dec_input = traj[:, 0]   # 初始输入
        # print(traj[0])
        # print (traj[:,0][0])
        # 教师强制 - 将目标词作为下一个输入
        # print(pad_lengths)
        lengths = (pad_lengths-np.ones_like(pad_lengths)*2).reshape(-1, 1)
        # print(traj)
        line = np.arange(batch_size).reshape(-1, 1)
        # print(lengths)
        index = np.hstack((line, lengths))
        # destination=tf.gather_nd(traj, index)
        # tf.print(tf.gather_nd(traj, index))
        # print (lengths)
        dec_hidden = tf.zeros(shape=(batch_size, dec_units), dtype=np.float64)
        cat_dec_hidden = tf.zeros(shape=(batch_size, cat_dec_units), dtype=np.float64)

        for t in range(1, traj.shape[1]):
            cat_predection, cat_dec_hidden = cat_decoder(dec_input, cat_embedding, cat_dec_hidden)
            cat_loss += cat_loss_function(traj[:, t], tf.nn.softmax(cat_predection))
            predictions, dec_hidden, output = decoder(dec_input, query_out, location_embeddings, A_hat, dec_hidden, cat_dec_hidden)
            loss += loss_function(traj[:, t], tf.nn.softmax((1-heuristic_prior_weight)*predictions + (heuristic_prior * heuristic_prior_weight)))
            total_loss = cat_loss + loss
            # log_des = mlp(output,query_out)
            # loss2=tf.keras.losses.SparseCategoricalCrossentropy(
            #     from_logits=True, reduction='none')(tf.expand_dims(destination,1), tf.nn.softmax(log_des))
            # loss2 = tf.reduce_mean(loss2)
            # loss = loss + loss2 * 0.1
            # tf.print(loss2)

            # print (tf.nn.softmax(predictions))
            # 使用教师强制
            # print(t)
            dec_input = traj[:, t]

    batch_loss = (loss / int(traj.shape[1]))
    variables = query.trainable_variables + decoder.trainable_variables + embedding_layer.trainable_variables + cat_decoder.variables
    # gradients = tape.gradient(loss, variables)
    gradients = tape.gradient(total_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


# --------------------------- pre-train_step -----------------------------
def pre_train_step(que, traj, real):
    '''
    训练步骤
    :param que: que.shape = (batch_size, 4)
    :param traj: traj.shape = (batch_size, traj_size)
    :param read: read.shape = (batch_size,)
    :param lr: learning rate
    :return: batch_loss
    '''
    loss = 0
    # optimizer
    pre_optimizer = tf.keras.optimizers.Adam(lr=0.001)
    with tf.GradientTape() as tape:
        location_embeddings, time_embeddings, cat_embedding = embedding_layer(dynamic_traning)
        query_out = query.pre_train(que, location_embeddings, time_embeddings)
        dec_input = traj[:, 0]  # 初始输入

        dec_hidden = tf.zeros(shape=(pre_batch_size, dec_units), dtype=np.float64)
        for t in range(1, traj.shape[1]):
            predictions, dec_hidden, output = decoder.pre_train(dec_input, query_out, location_embeddings, A_hat, dec_hidden)
            # 使用教师强制
            dec_input = traj[:, t]
        pre_prediction = pre_mlp(dec_hidden[0])
        real = tf.cast(real, dtype='float64')
        loss += tf.reduce_mean((pre_prediction - real) ** 2)

    batch_loss = (loss / int(traj.shape[1]))
    # batch_loss = (total_loss / int(traj.shape[1]))
    variables = query.trainable_variables + decoder.trainable_variables + embedding_layer.trainable_variables + pre_mlp.trainable_variables
    # gradients = tape.gradient(loss, variables)
    gradients = tape.gradient(loss, variables)
    pre_optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


# ------------------------------- evaluate -------------------------------
def evaluate(que, traj):
    #print('yes',traj)
    #traj=tf.expand_dims(traj,1)
    predict_traj = []
    location_embeddings, time_embeddings, cat_embedding = embedding_layer(dynamic_traning)
    query_out = query(que,location_embeddings, time_embeddings)
    dec_input = traj[:, 0]   # 初始输入
    #print('yes', traj.shape[1])
    # 将 GRU 输出值 predictions 作为下一个输入
    #print(traj.shape[1])
    realnum_poi=0
    # 获得预测poi个数
    for poi in traj[0]:
        if (poi == 0):
            break
        realnum_poi += 1
    realnum_poi = realnum_poi - 2

    start_poi = traj[:, 0]
    start_poi = tf.cast(start_poi, dtype=tf.int32)
    end_poi = traj[:, realnum_poi]
    dec_hidden = tf.zeros(shape=(1, dec_units), dtype=np.float64)
    cat_dec_hidden = tf.zeros(shape=(1, cat_dec_units), dtype=np.float64)
    # 添加门控table
    table = np.ones([poi_size], dtype=np.float64)
    table[start_poi[0]] = 0.
    table[end_poi[0]] = 0.
    #print(start_poi[0],end_poi[0],traj[0])
    table[vocab_to_int['END']] = 0. #end flag
    #print(table,'ss')
    #print(vocab_to_int['END'])
    for t in range(1, traj.shape[1]):
        # predictions, dec_hidden, _ = decoder(dec_input, query_out, location_embeddings, A_hat, dec_hidden)
        cat_predection, cat_dec_hidden = cat_decoder(dec_input, cat_embedding, cat_dec_hidden)
        predictions, dec_hidden, output = decoder(dec_input, query_out, location_embeddings, A_hat, dec_hidden, cat_dec_hidden)
        # 使用上一轮gru的输出predictions
        # print(predictions)
        # predictions = predictions + (heuristic_prior * heuristic_prior_weight)
        # dec_input.shape = (batch_size, )
        mask = tf.expand_dims(table, axis=0)
        dec_input = tf.argmax(tf.nn.softmax(predictions) * mask, 1)
        # dec_input = tf.argmax(tf.nn.softmax(predictions), 1)
        # print(dec_input.numpy()[0])
        table[dec_input.numpy()[0]] = 0.  # 防止预测出现重复poi
        predict_traj.append(dec_input)

    return predict_traj


def mytest(test_variables):
    encoder_test, decoder_test, test_batch_lenth, n_testTime, n_testDist1, n_testDist2, z_test, z_test_time, z_test_dist1, z_test_dist2 = test_variables
    step=0
    F1=[]
    pairs_F1=[]
    # print(encoder_test)
    while step < len(encoder_test) // batch_size:
        start_i = step * batch_size
        decode_batch = np.array(decoder_test[start_i:start_i + batch_size])
        pad_source_lengths = test_batch_lenth[start_i:start_i + batch_size]
        z_time = z_test_time[start_i:start_i + batch_size]
        z_in = z_test[start_i:start_i + batch_size]
        q_location = z_in
        q_time = z_time
        # print(q_time)
        # que = [[q_location[0]], [q_time[0]]]
        que = [[q_location[0]],[q_time[0]]]
        # print(q_location)
        # print(decode_batch)
        predict_traj=evaluate(que,np.expand_dims(decode_batch[0],0))
        # print('yes---',predict_traj)
        predict_traj=np.array(predict_traj)
        for v in range(1):
            length = pad_source_lengths[v] - 1
            actual = decode_batch[v][:length]
            # print('ac',actual)
            # print('v-',predict_traj[:,v][1:length - 1],predict_traj[:,v])
            recommend = np.concatenate([[actual[0]], predict_traj[:,v][0:length - 2]], axis=0)
            recommend = np.concatenate([recommend, [actual[-1]]], axis=0)
            # print('vs-', recommend)
            # print actual,recommend
            f = calc_F1(actual, recommend)
            p_f = calc_pairsF1(actual, recommend)
            F1.append(f)
            pairs_F1.append(p_f)
        step += 1
    return F1[0], pairs_F1[0], actual, recommend


if __name__ == "__main__":
    # print(vocab_to_int)
    # print(int_to_vocab)
    K = len(TRAIN_TRA)
    print('real data number@K', K)
    Tr_F1 = []
    Tr_pairsF1 = []
    Te_F1 = []
    Te_pairsF1 = []
    fTe_F1 = []
    fTe_pairsF1 = []
    L_F1 = []
    L_pairsF1 = []

    # pre_training for pre_decoder
    train_variables, test_variables = get_data(index=0, K=K)
    pre_data = data_augmentation1.DataAug(dat_suffix[dat_ix], vocab_to_int)
    # 通过Graph游走生成预训练数据 aug_trip
    # 通过真实数据生成预训练数据集 real_trip
    pre_data_set, pre_steps = pre_data.gen_pre_set(pre_batch_size, train_variables)

    total_loss = 0
    for i in range(K):
        # 重置模型参数
        # embedding_layer.reset_variable()
        # query.reset_variable()
        decoder.reset_variable()
        cat_decoder.reset_variable()

        print('Index of K', i)
        start_time = time.time()

        # 训练集数据 和 测试集数据
        train_variables, test_variables = get_data(index=i, K=K)
        encoder_train, decoder_train, train_batch_lenth, n_trainTime, n_trainDist1, n_trainDist2, z_train, z_train_time,\
        z_train_dist1, z_train_dist2 = train_variables
        # print(decoder_train[0])
        # print(z_train_time[0])
        # print(train_batch_lenth[0])


        PRE_EPOCHES = 20
        time_pre = time.time()
        for epoch in range(PRE_EPOCHES):
            for (batch, (que, traj, real)) in enumerate(pre_data_set.take(pre_steps)):
                batch_loss = pre_train_step(que, traj, real)
                total_loss += batch_loss
            print('pre-train loss:{}'.format(total_loss))
        print('pre-train end, take time {} sec'.format(time.time() - time_pre))


        EPOCHS = 20
        res = {}
        for epoch in range(EPOCHS):
            start = time.time()
            total_loss = 0
            step = 0
            while step < len(encoder_train) // batch_size:
                start_i = step * batch_size
                decode_batch = np.array(decoder_train[start_i:start_i + batch_size])
                pad_source_lengths = train_batch_lenth[start_i:start_i + batch_size]
                z_time = z_train_time[start_i:start_i + batch_size]
                z_in = z_train[start_i:start_i + batch_size]
                # print(len(z_train), len(z_train_time), len(decoder_train))
                # print('localtion:', z_train)
                # print('time:', z_train_time)
                q_location = z_in
                q_time = z_time
                # print(q_location)
                que = [q_location, q_time]
                # print('que:', que)
                # print('traj:', decoder_train)
                batch_loss = train_step(que, decode_batch, pad_source_lengths)
                total_loss += batch_loss
                step += 1
            if(epoch % 10 == 0):
                tf.print('loss', total_loss)
            # train_f1, train_pairs_f1, real, fake = test(train_variables)
            # print('EPOCH: {0}, train_f1: {1:.4f}, test_f1: {2:.4f}'.format(epoch, train_f1, train_pairs_f1))
            test_f1, test_pairs_f1, real, fake = mytest(test_variables)

            # max_epoch = -1
            res.setdefault(test_f1, []).append(test_pairs_f1)
        print('insex of K {0} take time {1} sec'.format(i, time.time() - start_time))

        # 记录实验结果
        keys = res.keys()
        keys = sorted(keys)
        print(keys[-1], max(res[keys[-1]]))
        if (max(res[keys[-1]])<1.0):
            print('real', real, 'fake', fake)
        L_F1.append(keys[-1])
        L_pairsF1.append(max(res[keys[-1]]))
    print('mode test F1', np.mean(L_F1), np.std(L_F1))
    print('model test pairsF1', np.mean(L_pairsF1), np.std(L_pairsF1))