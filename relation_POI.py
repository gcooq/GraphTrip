import numpy as np
import time
import scipy
import pandas as pd
import tensorflow as tf
from scipy.linalg import fractional_matrix_power
from sklearn.model_selection import train_test_split


class load_data():
    def __init__(self, city, mask=(0, 0, 0)):
        self.city = city
        self.int_to_vocab = None
        self.vocab_to_int = None
        self.mask = mask

    # calculate the distance between two places
    def calc_dist_vec(self, longitudes1, latitudes1, longitudes2, latitudes2):
        """Calculate the distance (unit: km) between two places on earth, vectorised"""
        # convert degrees to radians
        lng1 = np.radians(longitudes1)
        lat1 = np.radians(latitudes1)
        lng2 = np.radians(longitudes2)
        lat2 = np.radians(latitudes2)
        radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

        # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
        dlng = np.fabs(lng1 - lng2)
        dlat = np.fabs(lat1 - lat2)
        dist = 2 * radius * np.arcsin(np.sqrt(
            (np.sin(0.5 * dlat)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng)) ** 2))
        return dist

    def load_A(self):
        # =============================== datasets ====================================== #
        poi_name = "poi-" + self.city + ".csv"  # Edin
        tra_name = "traj-" + self.city + ".csv"
        embedding_name = self.city
        model = './logs/model_' + embedding_name + '.pkt'
        # =============================== data load ====================================== #
        # load original data
        op_tdata = open('origin_data/' + poi_name, 'r')
        ot_tdata = open('origin_data/' + tra_name, 'r')
        print('To Train', self.city)
        POIs = []
        Trajectory = []
        IDs = []
        for line in op_tdata.readlines():
            lineArr = line.split(',')
            temp_line = list()
            for item in lineArr:
                temp_line.append(item.strip('\n'))
            POIs.append(temp_line)
        POIs = POIs[1:]  # remove first line
        # print (POIs)
        # =============================== POI ====================================== #
        get_POIs = {}
        char_pois = []  # pois chars

        for items in POIs:
            char_pois.append(items[0])
            get_POIs.setdefault(items[0], []).append([items[2], items[3]])  # pois to category
        Users = []
        poi_count = {}
        for line in ot_tdata.readlines():
            lineArr = line.split(',')
            temp_line = list()
            if lineArr[0] == 'userID':
                continue
            poi_count.setdefault(lineArr[2], []).append(lineArr[2])
            IDs.append(lineArr[1])
            for i in range(len(lineArr)):
                if i == 0:
                    user = lineArr[i]
                    Users.append(user)  # add user id
                    temp_line.append(user)
                    continue
                temp_line.append(lineArr[i].strip('\n'))
            Trajectory.append(temp_line)
        Users = sorted(list(set(Users)))
        print('user number', len(Users))
        TRAIN_TRA = []
        TRAIN_USER = []
        TRAIN_TIME = []
        TRAIN_DIST = []
        DATA = {}  # temp_data
        print('original data numbers', len(Trajectory))

        # remove the trajectory which has less than three POIs
        for index in range(len(Trajectory)):
            if (int(Trajectory[index][-2]) >= 3):  # the length of the trajectory must over than 3
                DATA.setdefault(Trajectory[index][0] + '-' + Trajectory[index][1], []).append(
                    [Trajectory[index][2], Trajectory[index][3], Trajectory[index][4]])  # userID+trajID

        # print (len(IDs))
        IDs = set(IDs)
        print('original trajectory numbers', len(IDs))  # trajectory id

        for keys in DATA.keys():
            user_traj = DATA[keys]
            temp_poi = []
            temp_time = []
            temp_dist = []
            for i in range(len(user_traj)):
                temp_poi.append(user_traj[i][0])  # add poi id
                lon1 = float(get_POIs[user_traj[i][0]][0][0])
                lat1 = float(get_POIs[user_traj[i][0]][0][1])

                # start point
                lons = float(get_POIs[user_traj[0][0]][0][0])
                lats = float(get_POIs[user_traj[0][0]][0][1])

                # end point
                lone = float(get_POIs[user_traj[-1][0]][0][0])
                late = float(get_POIs[user_traj[-1][0]][0][1])

                sd = load_data.calc_dist_vec(self,lon1, lat1, lons, lats)
                ed = load_data.calc_dist_vec(self,lon1, lat1, lone, late)
                dt = time.strftime("%H:%M:%S", time.localtime(int(user_traj[i][1:][0])))
                # print dt.split(":")[0]
                temp_time.append(int(dt.split(":")[0]))  # add poi time
            TRAIN_USER.append(keys)
            TRAIN_TRA.append(temp_poi)
            TRAIN_TIME.append(temp_time)

        dictionary = {}
        for key in poi_count.keys():
            count = len(poi_count[key])
            dictionary[key] = count
        dictionary['END'] = 1
        new_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        print('trajectory with original POI id', TRAIN_TRA[0])
        print('poi number is', len(new_dict) - 1)  # three vitual POIs
        voc_poi = list()

        for item in new_dict:
            voc_poi.append(item[0])  # has been sorted by frequency
        print(len(voc_poi))

        # extract real POI id to a fake id
        def extract_words_vocab():
            int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
            vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
            return int_to_vocab, vocab_to_int

        int_to_vocab, vocab_to_int = extract_words_vocab()
        self.int_to_vocab, self.vocab_to_int = int_to_vocab, vocab_to_int
        print(self.int_to_vocab)
        print(self.vocab_to_int)

        # generate traning dataset without GO,END,PAD
        new_trainTs = list()
        for i in range(len(TRAIN_TRA)):  # TRAIN
            temp = list()
            for j in range(len(TRAIN_TRA[i])):
                temp.append(vocab_to_int[TRAIN_TRA[i][j]])
            new_trainTs.append(temp)
        print('trajectory with fake POI id', new_trainTs[0])

        Semantic_knowledge = np.zeros(shape=[len(voc_poi), len(voc_poi)], dtype=float)
        Spatial_knowledge = np.zeros(shape=[len(voc_poi), len(voc_poi)], dtype=float)
        Graph_mask = np.ones(shape=[len(voc_poi), len(voc_poi)], dtype=float)
        # semantic knowledge
        for tra in new_trainTs:
            for index in range(len(tra) - 1):
                first = tra[index] #visiting check-in and co-appear
                for index_2 in range(index+1,len(tra)):
                    second = tra[index_2]
                    Semantic_knowledge[first][second] = 1.
                    Semantic_knowledge[second][first] = 1.
        # semantic graph mask
        for i in range(Semantic_knowledge.shape[0]):
            mask_indict = np.random.choice(np.arange(Semantic_knowledge.shape[1]),
                                           size=int(Semantic_knowledge.shape[1] * self.mask[0]))
            Semantic_knowledge[i][mask_indict] = 0

        # add spatial knowledge
        for i in range(len(voc_poi)-1):
            if voc_poi[i+1] is not 'END':
                lon1=float(get_POIs[voc_poi[i]][0][0])
                lat1=float(get_POIs[voc_poi[i]][0][1])
                lon2=float(get_POIs[voc_poi[i+1]][0][0])
                lat2=float(get_POIs[voc_poi[i+1]][0][1])
                #print(load_data.calc_dist_vec(self, lon1, lat1, lon2, lat2))
                if (load_data.calc_dist_vec(self, lon1, lat1, lon2, lat2) <= 3):
                    Spatial_knowledge[vocab_to_int[voc_poi[i]]][vocab_to_int[voc_poi[i+1]]] = 1.
                    Spatial_knowledge[vocab_to_int[voc_poi[i+1]]][vocab_to_int[voc_poi[i]]] = 1.
        # spatial graph mask
        for i in range(Spatial_knowledge.shape[0]):
            mask_indict = np.random.choice(np.arange(Spatial_knowledge.shape[1]),
                                           size=int(Spatial_knowledge.shape[1] * self.mask[1]))
            Spatial_knowledge[i][mask_indict] = 0

        # first knowledge fusion
        Adj_matrix = Semantic_knowledge + Spatial_knowledge
        for i in range(Adj_matrix.shape[0]):
            for j in range(Adj_matrix.shape[1]):
                if Adj_matrix[i][j] >= 1:
                    Adj_matrix[i][j] = 1

        Adj_matrix = np.matrix(Adj_matrix)  # A
        # print(Adj_matrix)
        L = np.matrix(np.eye(Adj_matrix.shape[0]))
        print(Adj_matrix[0])
        Adj_matrix = Adj_matrix + L  # A_hat
        D = np.array(np.sum(Adj_matrix, axis=0))[0]
        D = np.matrix(np.diag(D))  # actually, it is D_hat
        A_hat = fractional_matrix_power(D, -0.5) * Adj_matrix * fractional_matrix_power(D, -0.5)

        return A_hat

    def load_poi_time(self):
        # add temporal information
        poi_time_list = []
        poi_time_graph = pd.read_csv('./poi-time/'+self.city + '-poi-time.csv', index_col=0)
        # temporal graph mask
        for i in range(poi_time_graph.shape[0]):
            mask_indict = np.random.choice(np.arange(poi_time_graph.shape[1]),
                                           size=int(poi_time_graph.shape[1] * self.mask[2]))
            poi_time_graph.iloc[i][mask_indict] = 0
        # print(poi_time_graph)
        for poiid in self.int_to_vocab.keys():
            voc = self.int_to_vocab[poiid]
            # print(voc)
            if(voc == 'END'):
                poi_time_list.append([0.0]*24)
            else:
                poi_time_list.append(poi_time_graph.loc[eval(voc)].tolist())
            # print(poiid, poi_time_graph.loc[eval(voc)].tolist())
        # [print(i) for i in poi_time_list]
        return poi_time_list

    def load_dataset_all(self, BATCH_SIZE):
        # 收集所有数据
        query_list = []
        trajs_list = []
        query_data = pd.read_csv('./traj_data/'+self.city+'-query.csv')
        trajs_data = open('./traj_data/'+self.city+'-trajs.dat', 'r')
        for i in query_data.index:
            query_item = query_data.loc[i].tolist()
            query_item[0] = self.vocab_to_int[str(query_item[0])]
            query_item[2] = self.vocab_to_int[str(query_item[2])]
            query_list.append(query_item)
        for line in trajs_data.readlines():
            tlist = [self.vocab_to_int[i] for i in line.split()]
            trajs_list.append(tlist)
        print('数据集总量：', len(query_list), len(trajs_list))
        trajs_list = tf.keras.preprocessing.sequence.pad_sequences(trajs_list,
                                                                   padding='post',
                                                                   value=self.vocab_to_int['END'])
        query_train = query_list
        trajs_train = trajs_list

        print('训练集：',len(query_train),len(trajs_train))

        dt_train = tf.data.Dataset.from_tensor_slices((query_train, trajs_train)).shuffle(len(query_train))
        dt_train = dt_train.batch(BATCH_SIZE, drop_remainder=True)
        # print(dt_train)

        return dt_train, int(len(query_train)/BATCH_SIZE)

    def load_dataset(self, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE):
        query_list = []
        trajs_list = []
        query_data = pd.read_csv('./traj_data/'+self.city+'-query.csv')
        trajs_data = open('./traj_data/'+self.city+'-trajs.dat', 'r')
        for i in query_data.index:
            query_item = query_data.loc[i].tolist()
            query_item[0] = self.vocab_to_int[str(query_item[0])]
            query_item[2] = self.vocab_to_int[str(query_item[2])]
            query_list.append(query_item)
        for line in trajs_data.readlines():
            tlist = [self.vocab_to_int[i] for i in line.split()]
            trajs_list.append(tlist)
        print('数据集总量：', len(query_list), len(trajs_list))
        trajs_list = tf.keras.preprocessing.sequence.pad_sequences(trajs_list,
                                                                   padding='post',
                                                                   value=self.vocab_to_int['END'])
        query_train, query_val, trajs_train, trajs_val = train_test_split(query_list, trajs_list, test_size=0.15)

        print('训练集：', len(query_train), len(trajs_train))
        print('测试集：', len(query_val), len(trajs_val))

        dt_train = tf.data.Dataset.from_tensor_slices((query_train, trajs_train)).shuffle(len(query_train))
        dt_train = dt_train.batch(TRAIN_BATCH_SIZE, drop_remainder=True)

        dt_val = tf.data.Dataset.from_tensor_slices((query_val, trajs_val)).shuffle(len(query_val))
        dt_val = dt_val.batch(VAL_BATCH_SIZE, drop_remainder=True)

        return dt_train, dt_val, int(len(query_train)/TRAIN_BATCH_SIZE), int(len(query_val)/VAL_BATCH_SIZE)

    def load_dataset_train(self, BATCH_SIZE):
        query_train = []
        trajs_train = []
        query_data = pd.read_csv('./traj_data/' + self.city + '-query-train.csv')
        trajs_data = open('./traj_data/' + self.city + '-trajs-train.dat', 'r')
        for i in query_data.index:
            query_item = query_data.loc[i].tolist()
            query_item[0] = self.vocab_to_int[str(query_item[0])]
            query_item[2] = self.vocab_to_int[str(query_item[2])]
            query_train.append(query_item)
        for line in trajs_data.readlines():
            tlist = [self.vocab_to_int[i] for i in line.split()]
            trajs_train.append(tlist)
        print('训练集总量：', len(query_train), len(trajs_train))
        trajs_train = tf.keras.preprocessing.sequence.pad_sequences(trajs_train,
                                                                    padding='post',
                                                                    value=self.vocab_to_int['END'])

        dt_train = tf.data.Dataset.from_tensor_slices((query_train, trajs_train)).shuffle(len(query_train))
        dt_train = dt_train.batch(BATCH_SIZE, drop_remainder=True)

        return dt_train, int(len(query_train) / BATCH_SIZE)

    def load_dataset_test(self, BATCH_SIZE):
        query_test = []
        trajs_test = []
        query_data = pd.read_csv('./traj_data/' + self.city + '-query-test.csv')
        trajs_data = open('./traj_data/' + self.city + '-trajs-test.dat', 'r')
        for i in query_data.index:
            query_item = query_data.loc[i].tolist()
            query_item[0] = self.vocab_to_int[str(query_item[0])]
            query_item[2] = self.vocab_to_int[str(query_item[2])]
            query_test.append(query_item)
        for line in trajs_data.readlines():
            tlist = [self.vocab_to_int[i] for i in line.split()]
            trajs_test.append(tlist)
        print('训练集总量：', len(query_test), len(trajs_test))
        trajs_test = tf.keras.preprocessing.sequence.pad_sequences(trajs_test, padding='post', value=self.vocab_to_int['END'])

        dt_test = tf.data.Dataset.from_tensor_slices((query_test, trajs_test)).shuffle(len(query_test))
        dt_test = dt_test.batch(BATCH_SIZE, drop_remainder=True)

        return dt_test, int(len(query_test) / BATCH_SIZE)


if __name__ == '__main__':
    A_data = load_data('Osak')
    A_hat = A_data.load_A()

    # 训练集数据 和 测试集数据
    dataset_train, steps_train = A_data.load_dataset_train(8)
    dataset_val, steps_val = A_data.load_dataset_test(4)

    for epoch in range(20):
        start = time.time()
        total_loss = 0
        # print('learning rate->>',lr)
        for (batch, (que, traj)) in enumerate(dataset_train.take(steps_train)):
            print('batch', batch)
            print('que:', que)
            print('tra:', traj)

        print('test dataset: ')
        for (batch, (que, traj)) in enumerate(dataset_val.take(steps_val)):
            print('batch', batch)
            print('que:', que)
            print('tra:', traj)





