import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from config import Config
CLS, SEP, PAD, MASK = 'CLS', 'SEP', 'PAD', '<mask>'

class BuildDataset(Dataset):
    def __init__(self, args, data_path, train=True):
        self.tokenizer = AutoTokenizer.from_pretrained('../roberta-base')
        self.arg1s, self.arg2s, self.labels = self.read_data(data_path, train)
        self.train = train
        self.args = args


    def __getitem__(self, item):
        return self.arg1s[item], self.arg2s[item], self.labels[item]

    def __len__(self):
        return len(self.arg1s)

    def pad(self, batch):
        batch_size = len(batch)
        max_length = 100
        arg_input_ids, arg_attention_mask, labels1, mask_idxs  = [], [], [], []

        for i in range(batch_size):
            token = CLS + batch[i][0] + MASK + batch[i][1] + SEP
            input = self.tokenizer(token, truncation=True, max_length=100, padding='max_length')
            arg_input_ids.append(input['input_ids'])
            arg_attention_mask.append(input['attention_mask'])


            arg1_len = len(batch[i][0].strip().split(' '))
            mask_idx = np.zeros(max_length)
            if arg1_len + 1 < max_length:
                mask_idx[batch[i][3] + 1] = 1
            else:
                mask_idx[0] = 1
            mask_idxs.append(mask_idx.tolist())

            labels1.append(batch[i][2])

            if self.args == 4:
                label_idx = np.zeros(4)
                label_idx[batch[i][2][0]] = 1
            else:
                # 这里仍存在问题,
                label_idx = np.zeros(11)
                label_idx[batch[i][2][1]] = 1



            # arg_input_id = batch[i][0].strip('[').strip(']').split(',')
            # arg_input_id = list(map(int, arg_input_id))
            # arg_input_ids.append(arg_input_id)
            #
            # attention_mask = batch[i][1].strip('[').strip(']').split(',')
            # attention_mask = list(map(int, attention_mask))
            # arg_attention_mask.append(attention_mask)

        arg_input_ids = torch.LongTensor(arg_input_ids)
        arg_attention_mask = torch.LongTensor(arg_attention_mask)
        labels1 = torch.LongTensor(labels1)
        mask_idxs = torch.LongTensor(mask_idxs)


        return arg_input_ids, arg_attention_mask, labels1, mask_idxs

    def merge(self, arg1, arg2):
        token = CLS + arg1 + MASK + arg2 + SEP
        input = self.tokenizer(token, truncation=True, max_length=100, padding='max_length')
        arg1_len = len(arg1.strip().split(' '))
        return str(input['input_ids']), str(input['attention_mask']), arg1_len

    def read_data(self, data_path, train):
        # contents, arg1_lens = [], []
        # data:usecols只保留了需要的列
        df = pd.read_csv(data_path, delimiter='\t')
        # print(df.columns)
        if train:
            return list(df['arg1']), list(df['arg2']), list(df['label'])
        else:
            return list(df['arg1']), list(df['arg2']), list(df['label1'])





        # df['input_ids'], df['attention_mask'], df['arg1_lens'] = df.apply(lambda row: self.merge(row['arg1'], row['arg2']), axis=1)


class Data3(Dataset):
    def __init__(self, args, path='../pdtb3_ji/', graph_path='label_graph.g'):
        super(Dataset, self).__init__()

        # kwargs = {'batch_size': args.per_gpu_train_batch_size * args.n_gpu, 'shuffle': args.shuffle, 'drop_last': False}
        kwargs = {'batch_size': 8, 'shuffle': True, 'drop_last': False}
        # config = Config(path, graph_path)
        # get_node_input(config)

        train_data = BuildDataset(path+'train.tsv')
        dev_data = BuildDataset(path+'dev.tsv', train=False)
        test_data = BuildDataset(path+'test.tsv', train=False)

        self.train_loader = DataLoader(train_data, **kwargs, collate_fn=train_data.pad)
        self.dev_loader = DataLoader(dev_data, **kwargs, collate_fn=dev_data.pad)
        self.test_loader = DataLoader(test_data, **kwargs, collate_fn=test_data.pad)

if __name__ == '__main__':
    d = Data()


