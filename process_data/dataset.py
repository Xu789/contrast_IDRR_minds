import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from collections import defaultdict
from config import Config

CLS, SEP, PAD, MASK = 'CLS', 'SEP', 'PAD', '<mask>'

class BuildDataset(Dataset):
    def __init__(self, data_path, config, args):
        super(Dataset, self).__init__()
        self.contents, self.arg1_lens = self.read_data(config, data_path)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('../roberta-base')
        self.args = args


    def __getitem__(self, item):
        return self.contents[item][0], self.contents[item][1], self.contents[item][2], self.contents[item][3], self.arg1_lens[item]

    def __len__(self):
        return len(self.contents)

    def pad(self, batch):
        batch_size = len(batch)
        max_length = len(batch[0][0])
        arg_input_ids, arg_attention_mask, labels1, mask_idxs, label_idxs = [], [], [], [], [], []

        for i in range(batch_size):
            arg_input_ids.append(batch[i][0])
            arg_attention_mask.append(batch[i][1])
            labels1.append(batch[i][2])
            # labels2.append(batch[i][3])
            # mask_idx, label_idx = np.zeros(max_length), np.zeros(max_length)
            mask_idx = np.zeros(max_length)

            # +1： 'CLS'
            if batch[i][4] + 1< max_length:
                mask_idx[batch[i][4] + 1] = 1
            else:
                mask_idx[0] = 1

            if self.args.n == 4:
                label_idx = np.zeros(4)
                label_idx[batch[i][2][0]] = 1

            else:
                label_idx = np.zeros(11)
                label_idx[batch[i][2][1]] = 1

            # label_idx[int(batch[i][2][0]) + 1] = 1
            mask_idxs.append(mask_idx.tolist())
            label_idxs.append(label_idx.tolist())

        arg_input_ids = torch.LongTensor(arg_input_ids)
        arg_attention_mask = torch.LongTensor(arg_attention_mask)
        labels1 = torch.LongTensor(labels1)
        # labels2 = torch.LongTensor(labels2)

        mask_idxs = torch.LongTensor(mask_idxs)
        label_idxs = torch.LongTensor(label_idxs)

        return arg_input_ids, arg_attention_mask, labels1, mask_idxs, label_idxs

    def read_data(self, config, data_path):
        contents = []
        arg1_lens = []
        label_lens = []
        with open(data_path, 'r', encoding='UTF-8') as f:
            for index, line in enumerate(f):
                line = line.strip()
                labels1, labels2, arg1, arg2, arg1_len = [_.strip() for _ in line.split("|||")]
                labels1, labels2 = eval(labels1), eval(labels2)

                labels1[0] = config.top2i[labels1[0]] if labels1[0] is not None else -1
                labels1[1] = config.sec2i[labels1[1]] if labels1[1] is not None else -1
                labels1[2] = config.conn2i[labels1[2]] if labels1[2] is not None else -1
                labels2[0] = config.top2i[labels2[0]] if labels2[0] is not None else -1
                labels2[1] = config.sec2i[labels2[1]] if labels2[1] is not None else -1
                labels2[2] = config.conn2i[labels2[2]] if labels2[2] is not None else -1

                # arg1_token = config.tokenizer.tokenize(arg1)
                # arg2_token = config.tokenizer.tokenize(arg2)
                token = CLS + arg1 + MASK + arg2 + SEP

                input = self.tokenizer(token, truncation=True, max_length=100, padding='max_length')
                input_ids, attention_mask = input['input_ids'], input['attention_mask']
                label1, label2 = [labels1[0], labels1[1], labels1[2]], [labels2[0], labels2[1], labels2[2]]

                contents.append([input_ids, attention_mask, label1, label2])
                arg1_lens.append(int(arg1_len))
        return contents, arg1_lens


def get_node_input(config):
    node_input = config.tokenizer(config.label, padding='longest', return_tensors='pt')
    torch.save(node_input, 'node_input')
    return


class Data(Dataset):
    def __init__(self, args, path='PDTB/Ji/data/', graph_path='label_graph.g'):
        super(Dataset, self).__init__()

        kwargs = {'batch_size': args.per_gpu_train_batch_size * args.n_gpu, 'shuffle': args.shuffle, 'drop_last': False}

        config = Config(path, graph_path)
        # get_node_input(config)

        train_data = BuildDataset(path+'train.txt', config, args)
        dev_data = BuildDataset(path+'dev.txt', config, args)
        test_data = BuildDataset(path+'test.txt', config, args)

        self.train_loader = DataLoader(train_data, **kwargs, collate_fn=train_data.pad)
        self.dev_loader = DataLoader(dev_data, **kwargs, collate_fn=dev_data.pad)
        self.test_loader = DataLoader(test_data, **kwargs, collate_fn=test_data.pad)
























"""
def get_i2label(config):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    label = config.i2top + config.i2sec + config.i2conn
    i2label = {xid: tokenizer.encode(x) for xid, x in enumerate(label)}
    torch.save(i2label, 'i2label.pt')

    label_graph = torch.load('label_graph.g')
    # label_to_i = dict((x, xid) for xid, x in enumerate(label))
    # i_to_label = dict((xid, x) for xid, x in enumerate(label))

    torch.save(i2label, "i2label.pt")
    hier_label = defaultdict(set)
    for index, val in enumerate(label_graph):
        if index < 15:
            for i, x in enumerate(val):
                if x != 0 and i>index:
                    hier_label[index].add(i)
    torch.save(hier_label, "hier_label.pt")
    return

    # # dictionary用来存储层次化标签
    # dictionary = {}
    # for index, val in enumerate(label_graph):
    #     node = i2label[index]
    #     if node not in dictionary:
    #         dictionary[node] = []
    #     else:
    #         for i, x in enumerate(val):
    #             if x != 0:
    #                 dictionary[node].append(i2label[i])

"""

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pgtbs', '--per_gpu_train_batch_size', type=int, default=32,
                        help='the batch size for per gpu device in the training..')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='choose the number of cuda')
    parser.add_argument('--shuffle', type=str, default=True,
                        help='whether to shuffle the data')
    args = parser.parse_args()

    data = Data(args, path='../PDTB/Ji/data/', graph_path='../label_graph.g')

    for i, batch in enumerate(data.train_loader):
        arg_input_ids, arg_attention_mask, label1, label2, arg1_lens = batch

        if i == len(data.train_loader)-1:
            print('batch_size = {}'.format(args.per_gpu_train_batch_size * args.n_gpu))
            print('In the train dataset..')
            print('the size of arg_input_ids:{}'.format(arg_input_ids.size()))
            print('the size of arg_attention_mask:{}'.format(arg_attention_mask.size()))

            print('label1:{}'.format(label1))
            print('label2:{}'.format(label2))
            print('the size of arg1_lens:{}'.format(arg1_lens.size()))
            break



# if __name__ == '__main__':
    # config = Config(data_path='../PDTB/Ji/data/', label_graph_path='../label_graph.g')
    # get_node_input(config)
    # test()