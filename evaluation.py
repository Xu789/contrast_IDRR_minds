
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import random
import torch
from config import Config
from collections import defaultdict
# sets = Config(args)

def compute_confusion_matrix(preds, label1, label2, n):
    """
    计算混淆矩阵，以及精确率
    精确率计算：TP/TP+FP，对于一个值的预测preds[i]，如果和label1[i]不同，但是和label2[i]相同，也算预测正确。
    @param preds:
    @param label1:
    @param label2:
    @param n:label的数目
    @return:
    """
    matrix = [[0] * n for _ in range(n)]
    batch_size = len(preds)
    for i in range(batch_size):
        p, l1, l2 = preds[i], label1[i], label2[i]
        # if p == l1:
        #     matrix[p][l1] += 1
        # else:
        # 如果l1与p不同[预测错误], 但是l2与p相同，我们也认为该类预测正确
        if p != l1 and p == l2:
            matrix[p][l2] += 1
            continue
        matrix[p][l1] += 1


    # precision_matrix：矩阵中的对应元素，每个类分类精确率
    precision_matrix = np.diagonal(matrix) / np.sum(matrix, axis=1)
    # precision：总体精确率
    precision = np.sum(precision_matrix) / len(precision_matrix)


    f1 = f1_score(y_true=label1, y_pred=preds, average='macro')
    return {'precision':precision,
            'f1':f1}


# def compute_acc_f1(fir_preds, fir_truth, sec_preds, sec_truth, conn_preds, conn_truth, args, config):
def compute_acc_f1(fir_preds, fir_truth, args, config):

    fir_f1 = f1_score(fir_truth, fir_preds, average=None)
    fir_acc = precision_score(fir_truth, fir_preds, average='micro'),
    fir_macro_f1 = np.mean(fir_f1)
    return fir_acc[0], fir_macro_f1

    """
    @param preds:
    @param label1:
    @param n: n分类预测任务
    @return:
    """

    # fir_f1_res, sec_f1_res, conn_f1_res = defaultdict(float), defaultdict(float), defaultdict(float)
    # # acc = precision_score(truth, preds, average=None)  #返回列表，列表中为每个类的精确度acc
    # fir_f1, sec_f1, conn_f1 = f1_score(fir_truth, fir_preds, average=None),\
    #                           f1_score(sec_truth, sec_preds, average=None),\
    #                           f1_score(conn_truth, conn_preds, average=None)
    # fir_acc, sec_acc, conn_acc = precision_score(fir_truth, fir_preds, average='micro'),\
    #                              precision_score(sec_truth, sec_preds, average='micro'),\
    #                              precision_score(conn_truth, conn_preds, average='micro')
    #
    # fir_macro_f1 = np.mean(fir_f1)
    # sec_marco_f1 = np.mean(sec_f1)
    # conn_marco_f1 = np.mean(conn_f1)
    #
    #
    #
    # for i in range(min(4, len(fir_preds))):
    #     fir_f1_res[config.fir_sense[i]] = fir_f1[i]
    #     # return  fir_acc, fir_macro_f1, fir_f1_res # dict, float
    # for i in range(min(11, len(sec_preds))):
    #     sec_f1_res[config.sec_sense[i]] = sec_f1[i]
    # for i in range(min(102, len(conn_preds))):
    #     conn_f1_res[config.con_sense[i]] = conn_f1_res[i]

    # return fir_acc, fir_macro_f1, sec_acc, sec_marco_f1, conn_acc, conn_marco_f1

    fir_f1 = f1_score(fir_truth, fir_preds, average=None)
    fir_acc = precision_score(fir_truth, fir_preds, average='micro'),
    fir_macro_f1 = np.mean(fir_f1)
    return fir_acc[0], fir_macro_f1





    # else:
    #     for i in range(min(args.n_class, len(preds))):
    #         f1_res[config.sec_sense[i]] = f1[i]
    #     return  micro_acc, macro_f1, f1_res

































