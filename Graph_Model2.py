# -*- coding:  UTF-8 -*-
import tensorflow as tf
import math
import numpy as np
import pandas as pd
# ----------------------------------- QueryModel ---------------------------------------
class EMModel(tf.keras.Model):
    def __init__(self, voc_size, time_size, POI_em_size, time_em_size, category_size, category_em_size, A_hat, poi_time):
        super(EMModel, self).__init__()
        self.POI_em_size = POI_em_size
        self.time_em_size = time_em_size
        self.voc_size = voc_size
        self.time_size = time_size
        self.cat_size = category_size
        self.cat_em_size = category_em_size
        self.embedings1 = A_hat
        self.embedings2 = poi_time
        # np.random.uniform(low=-1.0,high=1.0,size=(self.voc_size, self.POI_em_size))#)tf.Variable(tf.random.uniform([self.voc_size, self.POI_em_size], -1.0, 1.0))
        # self.weight = tf.Variable(tf.random.truncated_normal([self.voc_size, self.POI_em_size], stddev=1.0 / math.sqrt(self.POI_em_size),dtype=tf.float64))
        # self.bias = tf.Variable(tf.zeros([self.POI_em_size], dtype=tf.float64))
        self.cat_embedding = np.random.uniform(low=0.0, high=1.0, size=(self.cat_size, self.cat_em_size))
        # tf.random.uniform([self.time_em_size, self.time_em_size], -1.0, 1.0))
        # self.w = tf.Variable(tf.random.normal([24, self.voc_size], mean=0, stddev=1, dtype='float64'))
        self.fc1 = tf.keras.layers.Dense(self.POI_em_size, use_bias=False)
        self.fc2 = tf.keras.layers.Dense(self.POI_em_size, use_bias=False)

        self.fc3 = tf.keras.layers.Dense(self.time_em_size, use_bias=False)  # ,use_bias=False

    def call(self, is_dynamic=True):
        self.embedings1 = tf.cast(self.embedings1, dtype='float64')
        self.embedings2 = tf.cast(self.embedings2, dtype='float64')

        if(is_dynamic == True):
            # x = tf.matmul(self.embedings2, self.w)
            # out = tf.concat([x, self.embedings1], 1)
            # out = x + self.embedings1
            # location_embeddings = tf.nn.relu(self.fc1(out))
            # location_embeddings = tf.nn.tanh(self.fc1(out))
            # location_embeddings = self.fc2(location_embeddings)
            out1 = tf.nn.relu(self.fc1(self.embedings1))
            out2 = tf.nn.relu(self.fc2(self.embedings2))
            location_embeddings = out1 + out2
            # time_embeddings = tf.nn.relu(self.fc3(tf.transpose(self.embedings2)))
            time_embeddings = self.fc2.weights

            # 随机 time embedding
            # time_embeddings = tf.Variable(np.random.uniform(low=0.0, high=1.0, size=(24, self.time_em_size)))
            # one-hot 编码的time embedding
            # time_embeddings = tf.one_hot([i for i in range(24)], 24, dtype='float64')

        else:
            # location_embeddings = self.embedings
            # location_embeddings = np.random.uniform(low=0.0,high=1.0,size=(self.voc_size, self.POI_em_size))
            out = tf.concat([self.embedings1, self.embedings2], 1)
            location_embeddings = tf.nn.relu(self.fc1(out))

        return location_embeddings, time_embeddings, self.cat_embedding

    def reset_variable(self):
        self.fc1 = tf.keras.layers.Dense(self.POI_em_size)
        self.fc2 = tf.keras.layers.Dense(self.POI_em_size)


# ----------------------------------- QueryModel ---------------------------------------
class EMModel2(tf.keras.Model):
    def __init__(self,voc_size,time_size,POI_em_size,time_em_size,A_hat):
        super(EMModel2, self).__init__()
        self.POI_em_size = POI_em_size
        self.time_em_size = time_em_size
        self.voc_size=voc_size
        self.time_size=time_size
        self.embedings =A_hat
        # np.random.uniform(low=-1.0,high=1.0,size=(self.voc_size, self.POI_em_size))#)tf.Variable(tf.random.uniform([self.voc_size, self.POI_em_size], -1.0, 1.0))
        # self.weight = tf.Variable(tf.random.truncated_normal([self.voc_size, self.POI_em_size], stddev=1.0 / math.sqrt(self.POI_em_size),dtype=tf.float64))
        # self.bias = tf.Variable(tf.zeros([self.POI_em_size], dtype=tf.float64))
        self.time_embeddings = np.random.uniform(low=0.0,high=1.0,size=(self.time_size, self.time_em_size))#tf.random.uniform([self.time_em_size, self.time_em_size], -1.0, 1.0))
        self.fc1 = tf.keras.layers.Dense(self.POI_em_size)
        self.fc2 = tf.keras.layers.Dense(self.POI_em_size)

    def call(self,is_dynamic=True):
        time_embeddings = self.time_embeddings
        #print('time',self.time_embeddings)
        if(is_dynamic==True):
            # location_embeddings=tf.nn.tanh(self.fc1(self.embedings))#(tf.compat.v1.nn.xw_plus_b(self.embedings, self.weight, self.bias)) # tf.nn.relu
            location_embeddings = tf.nn.relu(self.fc1(self.embedings))
            # location_embeddings=self.fc2(location_embeddings)
        else:
            # location_embeddings = self.embedings
            # location_embeddings = np.random.uniform(low=0.0,high=1.0,size=(self.voc_size, self.POI_em_size))
            poi_list = []
            for i in range(self.voc_size):
                poi_list.append(np.random.randn(self.POI_em_size).tolist())
            location_embeddings = tf.constant(poi_list, dtype='float64')
        return location_embeddings, time_embeddings


class EMModel3():
    def __init__(self, city, voc_size, time_size, POI_em_size, time_em_size, category_size, category_em_size, em_type):
        self.city = city
        self.voc_size = voc_size
        self.time_size = time_size
        self.POI_em_size = POI_em_size
        self.time_em_size = time_em_size
        self.em_type = em_type
        self.cat_size = category_size
        self.cat_em_size = category_em_size

    def load_embedding(self, int_to_vocab):
        time_embeddings = np.random.uniform(low=0.0, high=1.0, size=(24, self.time_em_size))
        cat_embedding = np.random.uniform(low=0.0, high=1.0, size=(self.cat_size, self.cat_em_size))
        location_embeddings = []

        if self.em_type == 'deepwalk':
            pre = pd.read_csv('./embedding/' + self.city + '_embedding.csv', index_col=0)
            data = pd.read_csv('./embedding/' + self.city + '_deepwalk.csv')
            data.index = pre.index
            # print(data)

        if self.em_type == 'Random':
            location_embeddings = np.random.uniform(low=0.0, high=1.0, size=(self.voc_size, self.POI_em_size))
            location_embeddings = tf.cast(location_embeddings, dtype='float64')
            return location_embeddings, time_embeddings, cat_embedding

        if self.em_type == 'VGAE':
            data = pd.read_csv('./embedding/' + self.city + '_embedding.csv', index_col=0)

        if self.em_type == 'GAE':
            data = pd.read_csv('./embedding/' + self.city + '_embedding2.csv', index_col=0)

        if self.em_type == 'word2vec':
            pre = pd.read_csv('./embedding/' + self.city + '_embedding.csv', index_col=0)
            data = pd.read_csv('./embedding/' + self.city + '_tar_weight.csv')
            data.index = pre.index

        for poiint in int_to_vocab.keys():
            poi_vocab = int_to_vocab[poiint]
            if(poi_vocab == 'END'):
                continue
            poi_em = data.loc[eval(poi_vocab)].tolist()
            location_embeddings.append(poi_em)
        location_embeddings = tf.cast(location_embeddings, dtype='float64')

        return location_embeddings, time_embeddings, cat_embedding


class QueryModel(tf.keras.Model):
    def __init__(self, poi_dim,time_dim,K_dim):
        super(QueryModel, self).__init__()
        self.K_dim = K_dim
        self.poi_dim = poi_dim + time_dim
        self.w = tf.Variable(tf.random.normal([2*self.poi_dim, self.K_dim], mean=0, stddev=1,dtype='float64'))
        self.b = tf.Variable(tf.random.normal([self.K_dim], mean=0, stddev=1,dtype='float64'))

    def call(self, query_batch, poi_embedding, time_embedding, training=None, mask=None):
        query_location = np.array(query_batch[0])

        # tf.one_hot([i for i in range(24)], 24, dtype='float64') #we use the dense representation instead of one-hot
        # tf.print(query_location)
        # tf.print(query_location[:,-1])
        query_time = np.array(query_batch[1])
        start_poi = tf.nn.embedding_lookup(poi_embedding,query_location[:,0])
        end_poi = tf.nn.embedding_lookup(poi_embedding,query_location[:,1])
        start_time = tf.nn.embedding_lookup(time_embedding,query_time[:,0])
        end_time = tf.nn.embedding_lookup(time_embedding,query_time[:,1])
        # print(start_poi.shape)
        # print(start_time.shape)

        start_poi = tf.concat([start_poi,start_time],1)
        end_poi = tf.concat([end_poi, end_time], 1)
        X = tf.concat([start_poi,end_poi], 1)
        out = tf.matmul(X, self.w) + self.b

        return tf.nn.tanh(out)

    def pre_train(self, query_batch, poi_embedding, time_embedding, training=None, mask=None):

        start_poi = tf.nn.embedding_lookup(poi_embedding, query_batch[:, 0])
        end_poi = tf.nn.embedding_lookup(poi_embedding, query_batch[:, 2])
        start_time = tf.nn.embedding_lookup(time_embedding, query_batch[:, 1])
        end_time = tf.nn.embedding_lookup(time_embedding, query_batch[:, 3])

        start_poi = tf.concat([start_poi, start_time], 1)
        end_poi = tf.concat([end_poi, end_time], 1)
        X = tf.concat([start_poi, end_poi], 1)

        # print(X.shape, self.w.shape, self.b.shape)
        out = tf.matmul(X, self.w) + self.b

        return tf.math.tanh(out)

    def reset_variable(self):
        self.w = tf.Variable(tf.random.normal([2 * self.poi_dim, self.K_dim], mean=0, stddev=1, dtype='float64'))
        self.b = tf.Variable(tf.random.normal([self.K_dim], mean=0, stddev=1,dtype='float64'))


# using for train-split 20% and 80%
class QueryModel2(tf.keras.Model):
    def __init__(self, poi_dim, time_dim, K_dim):
        super(QueryModel2, self).__init__()
        self.K_dim = K_dim
        self.poi_dim = poi_dim + time_dim
        self.w = tf.Variable(tf.random.normal([2 * self.poi_dim, self.K_dim], mean=0, stddev=1, dtype='float64'))
        self.b = tf.Variable(tf.random.normal([self.K_dim], mean=0, stddev=1, dtype='float64'))

    def call(self, query_batch, poi_embedding, time_embedding, training=None, mask=None):
        poi_embedding = poi_embedding
        time_embdding = time_embedding

        start_poi = tf.nn.embedding_lookup(poi_embedding, query_batch[:, 0])
        end_poi = tf.nn.embedding_lookup(poi_embedding, query_batch[:, 2])
        start_time = tf.nn.embedding_lookup(time_embdding, query_batch[:, 1])
        end_time = tf.nn.embedding_lookup(time_embdding, query_batch[:, 3])

        start_poi = tf.concat([start_poi,start_time], 1)
        end_poi = tf.concat([end_poi, end_time], 1)
        X = tf.concat([start_poi,end_poi], 1)
        out = tf.matmul(X, self.w) + self.b

        # return shape(batch_size, K_dim)
        return tf.nn.tanh(out)

    def pre_train(self, query_batch, poi_embedding, time_embedding, training=None, mask=None):
        poi_embedding = poi_embedding
        time_embdding = time_embedding
        start_poi = tf.nn.embedding_lookup(poi_embedding, query_batch[:, 0])
        end_poi = tf.nn.embedding_lookup(poi_embedding, query_batch[:, 2])
        start_time = tf.nn.embedding_lookup(time_embdding, query_batch[:, 1])
        end_time = tf.nn.embedding_lookup(time_embdding, query_batch[:, 3])

        start_poi = tf.concat([start_poi, start_time], 1)
        end_poi = tf.concat([end_poi, end_time], 1)
        X = tf.concat([start_poi, end_poi], 1)
        out = tf.matmul(X, self.w) + self.b

        return tf.math.tanh(out)

    def reset_variable(self):
        self.w = tf.Variable(tf.random.normal([2 * self.poi_dim, self.K_dim], mean=0, stddev=1, dtype='float64'))
        self.b = tf.Variable(tf.random.normal([self.K_dim], mean=0, stddev=1,dtype='float64'))


# ----------------------------------- Decoder ---------------------------------------
class Decoder(tf.keras.Model):
    def __init__(self, poi_size, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.poi_size = poi_size
        # 用于注意力
        self.attention = BahdanauAttention(self.dec_units)
        # self.gru = tf.keras.layers.GRU(self.dec_units,
        #                                return_sequences=True,
        #                                return_state=True,
        #                                recurrent_initializer='glorot_uniform')
        self.gru = tf.keras.layers.GRUCell(self.dec_units)
        self.pre_fc = tf.keras.layers.Dense(self.poi_size)
        self.fc = tf.keras.layers.Dense(self.poi_size)
        self.fc_A = tf.keras.layers.Dense(self.dec_units)
        self.fc_B = tf.keras.layers.Dense(self.dec_units)

    def call(self, x, query, poi_embedding, A_hat, dec_hidden, cat_dec_hidden):
        embedding = poi_embedding
        x1 = tf.nn.embedding_lookup(embedding, x)
        x1 = tf.concat([x1, query], axis=1)

        output_, state = self.gru(x1, dec_hidden)
        context_vector, attention_weights = self.attention(output_, embedding)
        # output = tf.concat([context_vector, output_, cat_dec_hidden], axis=1)
        output = tf.concat([context_vector, output_, cat_dec_hidden[0]], axis=1)
        # remove attention
        # output = tf.concat([context_vector, output_, cat_dec_hidden[0]], axis=1)
        # remove cate_hidden
        # output = tf.concat([context_vector, output_,], axis=1)
        x = self.fc(output)

        return x, state, output_

    def pre_train(self, x, query, poi_embedding, A_hat, dec_hidden):
        embedding = poi_embedding
        x1 = tf.nn.embedding_lookup(embedding, x)
        x1 = tf.concat([x1, query], axis=1)

        output_, state = self.gru(x1, dec_hidden)
        context_vector, attention_weights = self.attention(output_, embedding)
        # output = tf.concat([context_vector, output_, cat_dec_hidden], axis=1)
        output = tf.concat([context_vector, output_], axis=1)
        x = self.pre_fc(output)

        return x, state, output_

    def reset_variable(self):
        self.attention = BahdanauAttention(self.dec_units)
        self.gru = tf.keras.layers.GRUCell(self.dec_units)
        self.fc = tf.keras.layers.Dense(self.poi_size)
        # self.fc_A = tf.keras.layers.Dense(self.dec_units)
        # self.fc_B = tf.keras.layers.Dense(self.dec_units)


# ----------------------------------- Category Decoder --------------------------------------
class Category_Decoder(tf.keras.Model):
    def __init__(self, category_size, poiint_categoryid_dict, dec_units):
        super(Category_Decoder, self).__init__()
        self.cat_size = category_size
        self.dec_units = dec_units
        self.poiint_categoryid_dict = poiint_categoryid_dict

        self.gru = tf.keras.layers.GRUCell(self.dec_units)
        self.fc = tf.keras.layers.Dense(category_size)

    def call(self, x, category_embedding, dec_hidden):
        embedding = category_embedding
        x = x.numpy()
        cat_list = []
        for i in x:
            if(i in self.poiint_categoryid_dict.keys()):
                cat_list.append(self.poiint_categoryid_dict[i])
            else:
                cat_list.append(0)
        x = tf.cast(cat_list, dtype='int32')
        x1 = tf.nn.embedding_lookup(embedding, x)
        output, state = self.gru(x1, dec_hidden)
        output = self.fc(output)

        return output, state

    def reset_variable(self):
        self.gru = tf.keras.layers.GRUCell(self.dec_units)
        self.fc = tf.keras.layers.Dense(self.cat_size)


class MLP(tf.keras.Model):
    def __init__(self,poi_size):
        super(MLP, self).__init__()
        self.fc = tf.keras.layers.Dense(poi_size)

    def call(self, output, query_out):

        x1 = self.fc(output)

        return x1


class PreMlp(tf.keras.Model):
    def __init__(self):
        super(PreMlp, self).__init__()
        self.fc = tf.keras.layers.Dense(1)

    def call(self, state):

        output = self.fc(state)

        return output


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # 隐藏层的形状 == （批大小，隐藏层大小）
        # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
        # 这样做是为了执行加法以计算分数
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 分数的形状 == （批大小，最大长度，1）
        # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
        # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
        attention_weights = tf.nn.softmax(score, axis=1)

        # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights