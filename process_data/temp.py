import torch
import pickle

from config import Config
from collections import defaultdict


def judge_symmetric():
    with open('../label_graph.g', 'rb') as f:
        label_graph = pickle.load(f)

    rows, cols = len(label_graph), len(label_graph[0])

    # 临接矩阵，对角线元素赋值
    for i in range(rows):
        if label_graph[i][i] == 0:
            label_graph[i][i] = 1

    for i in range(rows):
        for j in range(cols):
            if label_graph[i][j]==1:
                if label_graph[j][i]==0:
                    print('不是对称矩阵')
                    break
    print('是对称矩阵')
    with open('../label_graph.g', 'wb') as f:
        pickle.dump(label_graph, f)






def process():
    with open('../label_graph.g', 'rb') as f:
        label_graph = pickle.load(f)
    config = Config(data_path='../PDTB/Ji/data/', label_graph_path='../label_graph.g')
    label = config.label
    dictionary = dict()
    for i, l in enumerate(label):
        dictionary[i] = l
    print(dictionary)


    rows, cols = len(label_graph), len(label_graph[0])

    res = defaultdict(list)
    for i in range(rows):
        res1 = []
        for j in range(cols):
            if label_graph[i][j] == 1:
                res1.append(dictionary[j])
        res[dictionary[i]] = res1
    print(res)
    return




    # return dictionary




# def get_edge_idx():
#     with open('../label_graph.g', 'rb') as f:
#         label_graph = pickle.load(f)
#     edge_index = [[], []]
#     rows, cols = len(label_graph), len(label_graph[0])
#     for i in range(rows):
#         for j in range(cols):
#             if label_graph[i][j] == 1 and i < j:
#                 edge_index[0].append(i)
#                 edge_index[1].append(j)
#
#
#
#     edge_index = torch.LongTensor(edge_index)
#     torch.save(edge_index, 'edge_index')

if __name__ == '__main__':
    judge_symmetric()
