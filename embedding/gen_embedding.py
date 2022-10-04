#!/usr/bin/python3
# coding=utf-8
import numpy as np
import pandas as pd

file = open('./Edin_EMS_zero_vae.dat', 'r')
poi_list = []
pois_em = []
for line in file.readlines():
    line = line.split()
    if(len(line) <= 2):
        poi_size = eval(line[0])
        poi_em_size = eval(line[1])
    else:
        poi_em = [eval(i) for i in line[1:]]
        poi_list.append(eval(line[0]))
        pois_em.append(poi_em)

print('poi数目：',poi_size)
print('poi数目：',poi_em_size)

data = pd.DataFrame(pois_em)
data.index = poi_list
print(data)
data.to_csv('Edin_embedding.csv', index=True)

