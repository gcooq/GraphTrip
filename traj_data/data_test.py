#!/usr/bin/python3
# coding=utf-8

import pandas as pd
import numpy as np

# city = 'Osak'
# city = 'Glas'
# city = 'Toro'
# city = 'Edin'
city = 'Melb'
data = pd.read_csv(city+'-query.csv')

saving_list = []
test_index = []
data_num = len(data.index)
train_num = int(data_num * 0.85)
test_num = data_num - train_num
print('总数据{0}条，训练集{1}条，测试集{2}条'.format(data_num, train_num, test_num))
duplist = data.drop(['1','3'], axis=1, inplace=False)
duplist = duplist.duplicated(keep=False)

for i in duplist.index:
    if (duplist[i] == True):
        saving_list.append(i)
print('重复行：', saving_list)
print('重复行数：', len(saving_list))
# 训练集包含重复的query，测试集内的query和traj在训练集中未出现
# 处理query部分
query_saving = data.loc[saving_list]
# print(query_saving)
query_left = data.drop(saving_list, axis=0, inplace=False)
# print(query_left)
query_test = query_left.sample(n=test_num, replace=False, axis=0)
# print(query_test)
query_test.sort_index(axis=0,ascending=True,inplace=True)
# print(query_test)
test_index = query_test.index.tolist()
query_train = data.drop(test_index, axis=0, inplace=False)
print(len(query_train.index),len(query_test.index))

print('test index', test_index)
query_train.to_csv(city+'-query-train.csv', index=False)
query_test.to_csv(city+'-query-test.csv', index=False)

# 处理traj部分
file = open(city + '-trajs.dat', 'r')
file_train = open(city + '-trajs-train.dat', 'w')
file_test = open(city + '-trajs-test.dat', 'w')

for index, line in enumerate(file.readlines()):
    if (index in test_index):
        file_test.write(line)
    else:
        file_train.write(line)

file.close()
file_train.close()
file_test.close()