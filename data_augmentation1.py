#!/usr/bin/python3
# coding=utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import math

class DataAug():
    def __init__(self, city, vocab_to_int, walk_num=5, walk_length=2, negatives=3):
        '''
        PreQuery Model 构造函数
        :param dis_matric: 距离矩阵（反应两点的距离, 单位为 km）
        :param poi_num: 数据集poi个数
        :param walk_num: 每对结点生成的序列条数
        :param walk_length: 生成序列的最大深度
        '''
        self.dis_matric = pd.read_csv('./dis_matric/' + city + '_dis_matric.csv')
        self.poi_time = pd.read_csv('./poi-time/' + city + '-poi-time.csv', index_col=0)
        self.dis_matric.index = self.dis_matric.columns

        trajs_data = open('./traj_data/' + city + '-trajs.dat', 'r')
        self.train_data = []
        for line in trajs_data.readlines():
            tlist = [eval(i) for i in line.split()]
            self.train_data.append(tlist)

        self.poi_num = self.dis_matric.shape[0]
        self.vocab_to_int = vocab_to_int

        self.walk_num = walk_num
        self.walk_length = walk_length
        self.negatives = negatives
        self.poi_graph = None
        self.sentences = []  # 随机游走生成的序列

        self.pre_ques = []
        self.pre_trajs = []
        self.pre_reals = []

    def gen_sentences(self, start, end):
        for i in range(self.walk_num):
            sentence = [start, ]
            poi_now = start
            for j in range(self.walk_length):
                if np.sum(self.poi_graph.loc[poi_now]) != 1:  # 说明这个点与其他点都不相连
                    break
                poi_next = np.random.choice(self.poi_graph.columns, p=self.poi_graph.loc[poi_now])

                if poi_next == end:  # 得到结束点结束循环
                    sentence.append(poi_next)
                    break
                sentence.append(poi_next)
                poi_now = poi_next

            if (len(sentence) >= 3 and sentence[-1] == end):
                # sentence = [self.vocab_to_int[i] for i in sentence]
                # print(sentence)
                self.sentences.append(sentence)

    def gen_poi_time(self, poi):
        poi = eval(poi)
        if np.sum(self.poi_time.loc[poi]) != 1:
            return random.randint(0, 23)
        time = np.random.choice(self.poi_time.columns, p=self.poi_time.loc[poi])

        return time

    def pre_data_real_negative(self, real_traj):
        end_int = self.vocab_to_int['END']
        change_ints = list(self.vocab_to_int.values())
        change_ints.remove(end_int)
        site1 = 1
        if(end_int in real_traj):
            site2 = real_traj.index(end_int) - 2
        else:
            site2 = len(real_traj) - 2
        site = random.randint(site1, site2)
        change_int = random.sample(change_ints, 1)
        real_traj[site] = change_int[0]

        return real_traj

    def pre_data_aug_negative(self, aug_traj):
        end_int = self.vocab_to_int['END']
        change_ints = list(self.vocab_to_int.values())
        change_ints.remove(end_int)
        change_int = random.sample(change_ints, 1)
        aug_traj[1] = change_int[0]

        return aug_traj

    def pre_data_real(self, train_variables):
        # 通过真实轨迹，产生事实和反事实数据
        encoder_train, decoder_train, train_batch_lenth, n_trainTime, n_trainDist1, n_trainDist2, \
        z_train, z_train_time,z_train_dist1, z_train_dist2 = train_variables
        for i in range(len(decoder_train)):
            pre_que = z_train[i] + z_train_time[i]
            self.pre_ques.append(pre_que)
            self.pre_trajs.append(decoder_train[i])
            self.pre_reals.append(1.0)
            # 通过real traj 产生 negative traj
            for i in range(self.negatives):
                negative_traj = self.pre_data_real_negative(decoder_train[i])
                self.pre_ques.append(pre_que)
                self.pre_trajs.append(negative_traj)
                self.pre_reals.append(0.0)

    def gen_pre_set(self, batch_size, train_variables):
        self.poi_graph = pd.DataFrame(np.zeros(self.poi_num ** 2).reshape(self.poi_num, self.poi_num),
                                      index=self.dis_matric.index,
                                      columns=self.dis_matric.columns)
        for traj in self.train_data:
            for i in range(len(traj) - 1):
                self.poi_graph.loc[str(traj[i]), str(traj[i + 1])] += 1
        # print(self.poi_graph.index)
        # print(self.poi_graph)
        self.poi_graph = self.poi_graph + self.dis_matric

        for i in self.poi_graph.index:
            self.poi_graph.loc[i][i] = 0
        # print(self.poi_graph)
        for i in self.poi_graph.index:
            if np.sum(self.poi_graph.loc[i]) == 0:
                continue
            self.poi_graph.loc[i] = self.poi_graph.loc[i] / np.sum(self.poi_graph.loc[i])

        for start in self.poi_graph.index:
            for end in self.poi_graph.columns:
                if(start == end):
                    continue
                self.gen_sentences(start, end)

        # print('generate data ', len(self.sentences))

        for i in self.poi_time.index:
            if np.sum(self.poi_time.loc[i]) == 0:
                continue
            self.poi_time.loc[i] = self.poi_time.loc[i] / np.sum(self.poi_time.loc[i])
        self.poi_time.columns = np.arange(0, 24)

        query_list = []
        trajs_list = []
        real_list = []

        for sentence in self.sentences:
            flag = 1.0
            start_poi, end_poi = sentence[0], sentence[-1]
            query = [eval(start_poi), eval(end_poi), self.gen_poi_time(start_poi), self.gen_poi_time(end_poi)]
            query_list.append(query)
            # new_sentence = [self.vocab_to_int[i] for i in sentence]
            new_sentence = []
            for i in sentence:
                if i in self.vocab_to_int.keys():
                    new_sentence.append(self.vocab_to_int[i])
                else:
                    new_sentence.append(self.vocab_to_int['END'])
                    flag = 0.0
            trajs_list.append(new_sentence)
            real_list.append(flag)
            for i in range(self.negatives):
                aug_traj_negative = self.pre_data_aug_negative(new_sentence)
                query_list.append(query)
                trajs_list.append(aug_traj_negative)
                real_list.append(0.0)

        print('generate data : ', len(query_list), len(trajs_list), len(real_list))
        # print(trajs_list)

        # real trajs 生成 positive/negative data
        self.pre_data_real(train_variables)
        self.pre_ques += query_list
        self.pre_trajs += trajs_list
        self.pre_reals += real_list
        print('final pre train data: ', len(self.pre_ques), len(self.pre_trajs), len(self.pre_reals))
        # print(self.pre_trajs)
        # print(self.pre_ques)

        self.pre_trajs = tf.keras.preprocessing.sequence.pad_sequences(self.pre_trajs,
                                                                       padding='post',
                                                                       value=self.vocab_to_int['END'])
        pre_dataset = tf.data.Dataset.from_tensor_slices((self.pre_ques, self.pre_trajs, self.pre_reals)).shuffle(len(self.pre_ques))
        pre_dataset = pre_dataset.batch(batch_size, drop_remainder=True)

        return pre_dataset, int(len(self.pre_ques)/batch_size)

if __name__ == '__main__':
    city = 'Osak'
    poi_dis_matric = pd.read_csv('./poi-poi/' + city + '_poi_poi.csv')
    # print(poi_dis_matric.shape)
    poi_dis_matric.index = poi_dis_matric.columns
    poi_size = poi_dis_matric.shape[0]  # poi个数
    print('POI个数：',poi_size)
    dict1 = {'20': 0, '8': 1, '26': 2, '21': 3, '6': 4, '5': 5, '1': 6, '22': 7, '7': 8, '2': 9, '24': 10, '28': 11, '15': 12,'18': 13, '4': 14, '25': 15, '3': 16, '23': 17, '11': 18, '10': 19, '16': 20, '12': 21, '19': 22, '27': 23,'29': 24, '9': 25, '17': 26, 'END': 27}
    predata = DataAug(city, dict1)
    dataset, steps = predata.gen_pre_set(8)
