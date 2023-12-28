
import argparse
import sys
import pickle
import numpy as np
from collections import defaultdict

import mindspore

from pdtb2 import CorpusReader


class Processor():
    def __init__(self, **kwargs):
        self.splitting = kwargs.get('splitting', 1)
        self.data_path = kwargs.get('data_path', './pdtb2.csv')

        self.top_senses = set(['Temporal', 'Comparison', 'Contingency', 'Expansion'])
        self.selected_second_senses = set([
            'Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
            'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
            'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
            'Expansion.Alternative', 'Expansion.List'
        ])
        ## top_2_second用途: 若ConnHeadSemClass1, Conn2SemClass1只包含第一层级label时，用于推测最大概率的第二层级label
        self.top_2_second = {'Temporal': 'Temporal.Asynchronous',
                        'Comparison': 'Comparison.Contrast',
                        'Contingency': 'Contingency.Cause',
                        'Expansion': 'Expansion.Conjunction'}


        # splitting == 1:PDTB-Ji分割方式， splitting == 2:PDTB-Lin分割方式
        if self.splitting == 1:
            self.train_sec = ['02', '03', '04', '05',  '06', '07', '08', '09', '10', '11',
                         '12', '13', '14', '15', '16', '17', '18', '19', '20']
            self.dev_sec = ['00', '01']
            self.test_sec = ['21', '22']
            self.saved_path = '../PDTB/Ji/data/'
        if self.splitting == 2:
            self.train_sec = [
                '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
            ]
            self.dev_sec = ['22']
            self.test_sec = ['23']
            self.saved_path = '../PDTB/Lin/data/'


        # 用于计算ConnHeadSemClass1, Conn2SemClass1包含的label数目, 获取pdtb2.csv中实际label数目, 不包含预测得到的第二层级label
        # 训练测试验证集中label数目
        # self.top_label, self.sec_label, self.conn_label= defaultdict(int), defaultdict(int), defaultdict(int)
        self.train_num = {'top':defaultdict(int), 'sec':defaultdict(int), 'conn':defaultdict(int)}
        self.dev_num = {'top': defaultdict(int), 'sec': defaultdict(int), 'conn': defaultdict(int)}
        self.test_num = {'top': defaultdict(int), 'sec': defaultdict(int), 'conn': defaultdict(int)}



    # 将标签层次化处理
    def hierarchy_process(self):
        """
        将标签层次化处理：
        @return:
        """
        # sense1_train:[top, second, connective], sense2_train:[None, None, None]
        arg1_train, arg2_train, sense1_train, sense2_train = [], [], [], []
        arg1_dev, arg2_dev, sense1_dev, sense2_dev = [], [], [], []
        arg1_test, arg2_test, sense1_test, sense2_test = [], [], [], []
        arg1_train_len, arg1_dev_len, arg1_test_len = [], [], []

        # other instances, 当ConnHeadSemClass1.split('.')[0:2]只包含第一层级label时，我们需要加入概率最大的第二层级label
        arg1_train_other, arg2_train_other, sense1_train_other, sense2_train_other = [], [], [], []
        arg1_dev_other, arg2_dev_other, sense1_dev_other, sense2_dev_other = [], [], [], []
        arg1_test_other, arg2_test_other, sense1_test_other, sense2_test_other = [], [], [], []
        arg1_train_other_len, arg1_dev_other_len, arg1_test_other_len = [], [], []

        # num_of_conn1_1: 只包含第一层级语义类的conn1的个数
        num_of_conn1_1 = 0


        for corpus in CorpusReader(self.data_path).iter_data():
            if corpus.Relation != 'Implicit':
                continue

            sense_split = corpus.ConnHeadSemClass1.split('.')
            # sense_l2: 为conn1两个层级的label, 示例:['Expansion','Conjunction']
            sense_l2 = '.'.join(sense_split[0:2])

            if sense_l2 in self.selected_second_senses:

                # self.top_label[sense_split[0]] += 1
                # self.sec_label[sense_l2] += 1
                # self.conn_label[corpus.Conn1] += 1
                arg1, pos1 = self.arg_filter(corpus.arg1_pos(wn_format=True))
                arg2, pos2 = self.arg_filter(corpus.arg2_pos(wn_format=True))
                """
                如果该条语料在训练集中：执行arg1_train,arg2_train添加arg1与arg2操作，同时sense1_train添加三个层级的label, 
                sense2_train为None。
                如果该条语料在测试集中：arg1_dev,arg2_dev添加arg1与arg2操作，同时sense1_dev添加三个层级的label
                若在测试集中同理。
                """
                if corpus.Section in self.train_sec:
                    arg1_train.append(arg1)
                    arg1_train_len.append(len(arg1))
                    arg2_train.append(arg2)
                    sense1_train.append([sense_split[0], sense_l2, corpus.Conn1])
                    sense2_train.append([None, None, None])
                    self.train_num['top'][sense_split[0]] += 1
                    self.train_num['sec'][sense_l2] += 1
                    self.train_num['conn'][corpus.Conn1] += 1

                elif corpus.Section in self.dev_sec:
                    arg1_dev.append(arg1)
                    arg1_dev_len.append(len(arg1))
                    arg2_dev.append(arg2)
                    sense1_dev.append([sense_split[0], sense_l2, corpus.Conn1])
                    self.dev_num['top'][sense_split[0]] += 1
                    self.dev_num['sec'][sense_l2] += 1
                    self.dev_num['conn'][corpus.Conn1] += 1

                elif corpus.Section in self.test_sec:
                    arg1_test.append(arg1)
                    arg1_test_len.append(len(arg1))
                    arg2_test.append(arg2)
                    sense1_test.append([sense_split[0], sense_l2, corpus.Conn1])
                    self.test_num['top'][sense_split[0]] += 1
                    self.test_num['sec'][sense_l2] += 1
                    self.test_num['conn'][corpus.Conn1] += 1
                else:
                    continue

                """
                对该条语料的conn2进行判断不为空，则进行上述相同处理。
                如果conn2只表示了第一层级的label，那么我们对第二层级进行最大概率推测，第三层级为conn1，将其加入到arg_train_other列表中，
                若该条语料库conn1与conn2均存在，则添加了2条train[x]数据，1条dev[x]数据， 1条test[x]数据，每条数据格式:arg1_[x], arg2_[x], sense1_[x], sense2_[x]
                """
                if corpus.Conn2 is not None:
                    sense_split = corpus.Conn2SemClass1.split('.')
                    sense_l2 = '.'.join(sense_split[0:2])
                    if sense_l2 in self.selected_second_senses:
                        #
                        # self.top_label[sense_split[0]] += 1
                        # self.sec_label[sense_l2] += 1
                        # self.conn_label[corpus.Conn2] += 1

                        if corpus.Section in self.train_sec:
                            arg1_train.append(arg1)
                            arg1_train_len.append(len(arg1))
                            arg2_train.append(arg2)
                            sense1_train.append([sense_split[0], sense_l2, corpus.Conn2])
                            sense2_train.append([None, None, None])

                            self.train_num['top'][sense_split[0]] += 1
                            self.train_num['sec'][sense_l2] += 1
                            self.train_num['conn'][corpus.Conn2] += 1

                        elif corpus.Section in self.dev_sec:
                            sense2_dev.append([sense_split[0], sense_l2, corpus.Conn2])

                        elif corpus.Section in self.test_sec:
                            sense2_test.append([sense_split[0], sense_l2, corpus.Conn2])

                    # 如果conn2对应的语义类只表示了第一层级的label，我们将根据conn2对应的语义第一层类选择出现概率最大的第二层语义类
                    else:
                        sense_l2 = self.top_2_second[sense_split[0]]

                        # self.top_label[sense_split[0]] += 1
                        # # self.sec_label[sense_l2] += 1
                        # self.conn_label[corpus.Conn2] += 1

                        if corpus.Section in self.train_sec:
                            arg1_train_other.append(arg1)
                            arg1_train_other_len.append(len(arg1))
                            arg2_train_other.append(arg2)
                            sense1_train_other.append([sense_split[0], sense_l2, corpus.Conn2])
                            sense2_train_other.append([None, None, None])
                            self.train_num['top'][sense_split[0]] += 1
                            self.train_num['conn'][corpus.Conn2] += 1

                        elif corpus.Section in self.dev_sec:
                            arg1_dev_other.append(arg1)
                            arg1_dev_other_len.append(len(arg1))
                            arg2_dev_other.append(arg2)
                            sense1_dev_other.append([sense_split[0], sense_l2, corpus.Conn2])
                            self.dev_num['top'][sense_split[0]] += 1
                            self.dev_num['conn'][corpus.Conn2] += 1
                        elif corpus.Section in self.test_sec:
                            arg1_test_other.append(arg1)
                            arg1_test_other_len.append(len(arg1))
                            arg2_test_other.append(arg2)
                            sense1_test_other.append([sense_split[0], sense_l2, corpus.Conn2])
                            self.test_num['top'][sense_split[0]] += 1
                            self.test_num['conn'][corpus.Conn2] += 1
                        else:
                            continue
                else:
                    if corpus.Section in self.dev_sec:
                        sense2_dev.append([None, None, None])
                    elif corpus.Section in self.test_sec:
                        sense2_test.append([None, None, None])
            else:
                # ConnHeadSemClass1只包含第一层级语义类,
                num_of_conn1_1 += 1

                arg1, pos1 = self.arg_filter(corpus.arg1_pos(wn_format=True))
                arg2, pos2 = self.arg_filter(corpus.arg2_pos(wn_format=True))
                sense_l2 = self.top_2_second[sense_split[0]]

                # self.top_label[sense_split[0]] += 1
                # self.conn_label[corpus.Conn1] += 1


                if corpus.Section in self.train_sec:
                    arg1_train_other.append(arg1)
                    arg1_train_other_len.append(len(arg1))
                    arg2_train_other.append(arg2)
                    sense1_train_other.append([sense_split[0], sense_l2, corpus.Conn1])
                    sense2_train_other.append([None, None, None])
                    self.train_num['top'][sense_split[0]] += 1
                    self.train_num['conn'][corpus.Conn1] += 1

                elif corpus.Section in self.dev_sec:
                    arg1_dev_other.append(arg1)
                    arg1_dev_other_len.append(len(arg1))
                    arg2_dev_other.append(arg2)
                    sense1_dev_other.append([sense_split[0], sense_l2, corpus.Conn1])
                    self.dev_num['top'][sense_split[0]] += 1
                    self.dev_num['conn'][corpus.Conn1] += 1
                elif corpus.Section in self.test_sec:
                    arg1_test_other.append(arg1)
                    arg1_test_other_len.append(len(arg1))
                    arg2_test_other.append(arg2)
                    sense1_test_other.append([sense_split[0], sense_l2, corpus.Conn1])
                    self.test_num['top'][sense_split[0]] += 1
                    self.test_num['conn'][corpus.Conn1] += 1
                else:
                    continue

                if corpus.Conn2 is not None:
                    sense_split = corpus.Conn2SemClass1.split('.')
                    sense_l2 = '.'.join(sense_split[0:2])
                    sense_l2 = sense_l2 if sense_l2 in self.selected_second_senses else self.top_2_second[sense_split[0]]

                    # self.top_label[sense_split[0]] += 1
                    # if len(sense_split) > 1:
                    #     self.sec_label[sense_l2] += 1
                    # self.conn_label[corpus.Conn2] += 1


                    if corpus.Section in self.train_sec:
                        arg1_train_other.append(arg1)
                        arg1_train_other_len.append(len(arg1))
                        arg2_train_other.append(arg2)
                        sense1_train_other.append([sense_split[0], sense_l2, corpus.Conn2])
                        sense2_train_other.append([None, None, None])
                        self.train_num['top'][sense_split[0]] += 1
                        self.train_num['conn'][corpus.Conn2] += 1
                        if len(sense_split) > 1:
                            self.train_num['sec'][sense_l2] += 1

                    elif corpus.Section in self.dev_sec:
                        sense2_dev_other.append([sense_split[0], sense_l2, corpus.Conn2])


                    elif corpus.Section in self.test_sec:
                        sense2_test_other.append([sense_split[0], sense_l2, corpus.Conn2])

                else:
                    if corpus.Section in self.dev_sec:
                        sense2_dev_other.append([None, None, None])
                    elif corpus.Section in self.test_sec:
                        sense2_test_other.append([None, None, None])


        # combined two parts of data
        arg1_train.extend(arg1_train_other)
        arg1_train_len.extend(arg1_train_other_len)
        arg2_train.extend(arg2_train_other)
        sense1_train.extend(sense1_train_other)
        sense2_train.extend(sense2_train_other)

        arg1_dev.extend(arg1_dev_other)
        arg1_dev_len.extend(arg1_dev_other_len)
        arg2_dev.extend(arg2_dev_other)
        sense1_dev.extend(sense1_dev_other)
        sense2_dev.extend(sense2_dev_other)

        arg1_test.extend(arg1_test_other)
        arg1_test_len.extend(arg1_test_other_len)
        arg2_test.extend(arg2_test_other)
        sense1_test.extend(sense1_test_other)
        sense2_test.extend(sense2_test_other)


        assert len(arg1_train) == len(arg2_train) == len(sense1_train) == len(sense2_train) == len(arg1_train_len)
        assert len(arg1_dev) == len(arg2_dev) == len(sense1_dev) == len(sense2_dev) == len(arg1_dev_len)
        assert len(arg1_test) == len(arg2_test) == len(sense1_test) == len(sense2_test) == len(arg1_test_len)


        print('train size:', len(arg1_train))
        print('dev size:', len(arg1_dev))
        print('test size:', len(arg1_test))

        mode = {0:'train.txt', 1:'dev.txt', 2:'test.txt'}

        for i in range(3):
            file_name = self.saved_path + mode[i]
            if i == 0:
                self.write2txt(file_name, arg1_train, arg2_train, sense1_train, sense2_train, arg1_train_len)
            if i == 1:
                self.write2txt(file_name, arg1_dev, arg2_dev, sense1_dev, sense2_dev, arg1_dev_len)
            if i == 2:
                self.write2txt(file_name, arg1_test, arg2_test, sense1_test, sense2_test, arg1_test_len)

    def write2txt(self, file_name, arg1s, arg2s, sense1s, sense2s, arg1_len):
        with open(file_name, 'w') as f:
            for arg1, arg2, sense1, sense2, length in zip(arg1s, arg2s, sense1s, sense2s, arg1_len):
                print('{} ||| {} ||| {} ||| {} ||| {}'.format(sense1, sense2, ' '.join(arg1), ' '.join(arg2), length), file=f)


    def print_labels(self):
        """
        先执行self.hierarchy_process，然后运行此compute_labels函数有意义
        计算了ConnHeadSemClass1，Conn2SemClass1的label数
        优化：
        输出每个层级 train、dev、test的label数目
        定义数据结构： top_label defaultdict(int)
        result = 【train，dev，test】 train.append(top_label, sec_label, conn_label)
        @return:
        """
        print('训练集中标签数目')
        print(self.train_num)
        print('验证集中标签数目')
        print(self.dev_num)
        print('测试集中标签数目')
        print(self.test_num)



    def arg_filter(self, input):
        arg = []
        pos = []
        for w in input:
            if w[1].find('-') == -1:
                arg.append(w[0].replace('\/', '/'))
                pos.append(w[1])
        return arg, pos


def get_edge_idx():
    with open('../label_graph.g', 'rb') as f:
        label_graph = pickle.load(f)
    edge_index = [[], []]
    rows, cols = len(label_graph), len(label_graph[0])
    for i in range(rows):
        for j in range(cols):
            if label_graph[i][j] == 1:
                edge_index[0].append(i)
                edge_index[1].append(j)
    edge_index = torch.LongTensor(edge_index)
    torch.save(edge_index, 'edge_index')







if __name__ == '__main__':
    # get_edge_idx()
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', dest='func', choices=['pre', 'test'], type=str, default='pre')
    parser.add_argument('-s', dest='splitting', choices=[1, 2], type=int, default='1')   # 1 for 'Ji', 2 for 'Lin'、
    args = parser.parse_args()

    x = Processor(**vars(args))
    x.hierarchy_process()
    x.print_labels()




