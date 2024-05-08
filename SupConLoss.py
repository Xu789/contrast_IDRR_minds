import mindspore
import mindspore.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from mindspore import context


class Instance2label(nn.Cell):
    def __init__(self, args, temperature=0.5):
        super(Instance2label, self).__init__()
        self.temperature = temperature
        self.args = args
        self.l2_normalize = mindspore.ops.L2Normalize(axis=1)
    def construct(self, instance, label_embeds, connlabel):
        """
        @param instance: [batch, 768]
        @param label_embeds: [117, 768]
        @param connlabel: [batch]
        @return:
        """
        instance = self.l2_normalize(instance)
        label_embeds = self.l2_normalize(label_embeds)
        batch_size = instance.shape[0]







        if self.args.pdtb_version == 2:
            instance_dot_label = mindspore.ops.mm(instance, label_embeds[15:].T) / self.temperature
            conn_pos_mask = np.zeros((batch_size, 102))
        else:
            instance_dot_label = mindspore.ops.mm(instance, label_embeds[18:].T) / self.temperature
            conn_pos_mask = np.zeros((batch_size, 181))
        for i in range(batch_size):
            conn_pos_mask[i][connlabel[i]] = 1
        conn_pos_mask = mindspore.tensor(conn_pos_mask)

        logits_max, _ = mindspore.ops.max(instance_dot_label, axis=1, keepdims=True)
        logits = instance_dot_label - logits_max

        exp_logits = mindspore.ops.exp(logits)  # exp_logits: [batch, label_num]
        denominator = mindspore.ops.log(exp_logits.sum(axis=1) + 1e-12)
        molecule = mindspore.ops.log(mindspore.ops.mul(exp_logits, conn_pos_mask).sum(axis=1))
        log_prob = molecule - denominator
        return -log_prob.sum(axis=0) / batch_size

class Label2label(nn.Cell):
    def __init__(self, args, temperature=0.5):
        super(Label2label, self).__init__()
        self.temperature = temperature
        self.args = args
        self.zero = mindspore.tensor(0)
        self.zero.requires_grad = False
        self.l2_normalize = mindspore.ops.L2Normalize(axis=1)
    def del_tensor_elem(self, x, index):
        """
        @param x: 删除tensor_x中位置为index的索引值，x: [batch, 117, 768]
        @param index: [batch]
        @return: [batch, 101, 768]
        """
        for i in range(x.size(0)):
            temp = mindspore.ops.cat((x[i][15:index[i]+15], x[i][index[i]+15+1:]), axis=0).unsqueeze(0)
            if i == 0:
                res = temp
            else:
                res = mindspore.ops.cat((res, temp), axis=0)

        return res
    def construct(self, label_embeds, firlabel, seclabel, connlabel):
        """
        @param label_embeds: [117, 768]
        @param firlabel: [batch]
        @param seclabel: [batch]
        @param connlabel: [batch]
        @return: loss
        """

        label_embeds = self.l2_normalize(label_embeds)
        batch_size = firlabel.shape[0]
        fir_labels, sec_labels, conn_labels = None, None, None

        # indices = torch.cat((firlabel, seclabel, connlabel), dim=0)
        # indices = indices.transpose(0, 1) # indices:[batch, 3]，生成的层次化标签索引
        for j in range(3):
            res = None
            for i in range(batch_size):
                if j == 0:
                    temp = mindspore.ops.index_select(label_embeds, 0, firlabel.unsqueeze(-1)[i])
                elif j == 1:
                    temp = mindspore.ops.index_select(label_embeds, 0, seclabel.unsqueeze(-1)[i])
                else:
                    temp = mindspore.ops.index_select(label_embeds, 0, connlabel.unsqueeze(-1)[i])
                if i == 0:
                    res = temp
                if i != 0:
                    res = mindspore.ops.cat((res, temp), axis=0)
            if j == 0:
                fir_labels = res  # fir_label: [batch, 768]
            if j == 1:
                sec_labels = res
            if j == 2:
                conn_labels = res #

        if self.args.pdtb_version == 2:
            batch_conn_labels = mindspore.ops.tile(label_embeds.unsqueeze(0), (batch_size,1,1))[:, 15: ,:]  # batch_conn_labels: [batch, 102, 768]
        else:
            batch_conn_labels = mindspore.ops.tile(label_embeds.unsqueeze(0), (batch_size,1,1))[:, 18: ,:]
        # molecule_conn_sec， molecule_conn_fir：；论文中对比损失函数的分子部分， [batch]

        ##归一化
        conn_max, _ = mindspore.ops.max(conn_labels, axis=1, keepdims=True)
        conn_labels_ = conn_labels - conn_max
        sec_max, _ = mindspore.ops.max(sec_labels, axis=1, keepdims=True)
        sec_labels_ = sec_labels - sec_max
        fir_max, _ = mindspore.ops.max(fir_labels, axis=1, keepdims=True)
        fir_labels_ = fir_labels - fir_max

        molecule_conn_sec = mindspore.ops.mul(conn_labels_, sec_labels_).sum(axis=1) / self.temperature
        molecule_conn_fir = mindspore.ops.mul(conn_labels_, fir_labels_).sum(axis=1) / self.temperature
        molecule_conn_sec = mindspore.ops.log(molecule_conn_sec)
        molecule_conn_fir = mindspore.ops.log(molecule_conn_fir)
        # molecule_conn_sec, molecule_conn_fir = torch.log(molecule_conn_sec), torch.log(molecule_conn_fir)
        # neg_conn_labels = self.del_tensor_elem(batch_conn_labels, connlabel)  # neg_conn_labels: [batch, 101, 768]
        denominator = mindspore.ops.bmm(conn_labels_.unsqueeze(1), batch_conn_labels.transpose(0,2,1)) # denominator: [batch, 1, 102]

        denominator = mindspore.ops.exp(denominator)
        denominator = denominator.squeeze(1).sum(axis=1) / self.temperature # denominator: [batch]
        denominator = mindspore.ops.log(denominator)
        pair_conn_sec = -(molecule_conn_sec - denominator)
        pair_conn_fir = -(molecule_conn_fir - denominator)
        loss = mindspore.ops.max(self.zero,(pair_conn_sec-pair_conn_fir).sum(axis=0) / batch_size)[0]

        return loss
