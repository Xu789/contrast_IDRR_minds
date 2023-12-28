import mindspore
import mindspore.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from mindspore import context

class InstanceAnchor(nn.Cell):
    def __init__(self, temp):
        super(InstanceAnchor, self).__init__()
        # self.alpha = alpha
        self.temp = temp

    def forward(self, instance, label_embeds, firlabel, seclabel, connlabel):
        # instance: [batch, hidden_state], label_embeds: [label_num, hidden_state], label: [batch]
        # device = (torch.device('cuda')
        #           if instance.is_cuda
        #           else torch.device('cpu'))
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        instance = F.normalize(instance, dim=-1)
        label_embeds = F.normalize(label_embeds, dim=-1)
        batch_size = instance.shape(0)
        label_num = label_embeds.shape(0)

        fir_pos_mask, sec_pos_mask, conn_pos_mask = np.zeros((batch_size, 4)), \
                                                    np.zeros((batch_size, 11)),\
                                                    np.zeros((batch_size, 102))
        for i in range(batch_size):
            fir_pos_mask[i][firlabel[i]] = 1
            sec_pos_mask[i][seclabel[i]] = 1
            conn_pos_mask[i][connlabel[i]] = 1

        fir_pos_mask = mindspore.tensor(fir_pos_mask)
        sec_pos_mask = mindspore.tensor(sec_pos_mask)
        conn_pos_mask = mindspore.tensor(conn_pos_mask)

        loss = 0
        for l in range(3):
            if l==0:
                instance_dot_label = mindspore.ops.mm(instance, label_embeds[0:4].T) / self.temp
                label, pos_mask = firlabel, fir_pos_mask

            elif l==1:
                instance_dot_label = mindspore.ops.mm(instance, label_embeds[4:15].T) / self.temp
                label, pos_mask = seclabel, sec_pos_mask
            else:
                instance_dot_label = mindspore.ops.mm(instance, label_embeds[15:].T) / self.temp
                label, pos_mask = connlabel, conn_pos_mask

            logits_max, _ = mindspore.ops.max(instance_dot_label, axis =1, keepdims=True)
            logits = instance_dot_label - logits_max.data
            exp_logits = mindspore.ops.exp(logits)  # exp_logits: [batch, label_num]
            denominator = mindspore.ops.log(exp_logits.sum(dim=1) + 1e-12)
            molecule = mindspore.ops.log((exp_logits * pos_mask).sum(dim=1))
            log_prob = molecule - denominator
            hie_loss = -log_prob.mean()

            loss += hie_loss

        return loss / 3

class LabelAnchor(nn.Cell):
    def __init__(self, temp):
        super(LabelAnchor, self).__init__()
        # self.alpha = alpha
        self.temp = temp

    def forward(self, instance, label_embeds, firlabel, seclabel, connlabel):
        # instance: [batch, hidden_state], label_embeds: [label_num, hidden_state], label: [batch]
        # device = (torch.device('cuda')
        #           if instance.is_cuda
        #           else torch.device('cpu'))
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        instance = F.normalize(instance, dim=-1)
        label_embeds = F.normalize(label_embeds, dim=-1)
        batch_size = instance.size(0)
        label_num = label_embeds.size(0)


        fir_pos_mask, sec_pos_mask, conn_pos_mask = np.zeros((4, batch_size)),\
                                                    np.zeros((11, batch_size)),\
                                                    np.zeros((102, batch_size))

        for i in range(4):
            for j in range(batch_size):
                if firlabel[j] == i:
                    fir_pos_mask[i][j] = 1
        for i in range(11):
            for j in range(batch_size):
                if seclabel[j] == i:
                    sec_pos_mask[i][j] = 1
        for i in range(102):
            for j in range(batch_size):
                if connlabel[j] == i:
                    conn_pos_mask[i][j] = 1

        fir_pos_mask, sec_pos_mask, conn_pos_mask = mindspore.tensor(fir_pos_mask),\
                                                    mindspore.tensor(sec_pos_mask),\
                                                    mindspore.tensor(conn_pos_mask)


        loss = 0
        for l in range(3):
            if l==0:
                label_dot_instance = mindspore.ops.mm(label_embeds[0:4], instance.T) / self.temp
                label, pos_mask = firlabel, fir_pos_mask
            elif l==1:
                label_dot_instance = mindspore.ops.mm(label_embeds[4:15], instance.T) / self.temp
                label, pos_mask = seclabel, sec_pos_mask
            else:
                label_dot_instance = mindspore.ops.mm(label_embeds[15:], instance.T) / self.temp
                label, pos_mask = connlabel, conn_pos_mask

            logits_max, _ = mindspore.ops.max(label_dot_instance, dim=1, keepdim=True)
            logits = label_dot_instance - logits_max.detach()
            exp_logits = mindspore.ops.exp(logits)  # exp_logits: [label_num, batch]

            denominator = mindspore.ops.log(exp_logits.sum(dim=1) + 1e-12)
            # denominator:[label_num]
            # molecule = torch.log((exp_logits * positive_mask).sum(dim=1))
            # molecule : [label_num]
            wolog_molecule = (exp_logits * pos_mask).sum(dim=-1)

            # num_positives_per_row = copy.deepcopy(wolog_molecule)
            # wolog_molecule = torch.where(torch.isinf(wolog_molecule), torch.full_like(wolog_molecule, 1), wolog_molecule)
            molecule = mindspore.ops.log(wolog_molecule)
            log_prob = molecule - denominator
            # log_prob：[label_num]
            hie_loss = - log_prob[wolog_molecule > 0].mean()
            loss += hie_loss



        return loss / 3




