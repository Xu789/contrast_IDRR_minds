import pandas as pd
import numpy as np

import mindspore
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import RobertaTokenizer
CLS, SEP, PAD, MASK = 'CLS', 'SEP', 'PAD', '<mask>'

"""
PDTB2.0
训练集：将SCLASS1A，SCLASS2A，考虑在内
验证集测试集：判断是否跟SCLASS1A标签一致

PDTB3.0
目前：训练集：将SCLASS1A，SCLASS1B，SCLASS2A，SCLASS2B均考虑在内
测试集验证集：判断是否跟SCLASS1A标签一致
最终：训练集：将SCLASS1A，SCLASS2A考虑在内，验证集测试集：判断是否跟SCLASS1A标签一致
"""

class BasePdtbDataset:
    def __init__(self, args, **kwargs):

        """
        @param args:
        """
        super(BasePdtbDataset, self).__init__()

        data_path = kwargs.get('data_path', args.data_path)
        self.mode = kwargs.get('mode', 'train')
        usecols = kwargs.get('usecols', None)

        self.args = args
        # 数据集版本定义
        self.fir_sense = ['Temporal', 'Contingency', 'Comparison', 'Expansion']
        self.fir_sense2i = dict((s,i) for i, s in enumerate(self.fir_sense))
        if args.pdtb_version == 2:
            self.sec_sense = [
                'Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
                'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
                'Expansion.Alternative', 'Expansion.List'
            ]
            self.sec_sense2i = dict((s,i) for i, s in enumerate(list(self.sec_sense)))
            self.top2second = {'Temporal': 'Temporal.Asynchronous',
                               'Comparison': 'Comparison.Contrast',
                               'Contingency': 'Contingency.Cause',
                               'Expansion': 'Expansion.Conjunction'}
            self.conn = ['however', 'and', 'for example', 'although', 'in short', 'rather',
                               'specifically', 'then', 'also', 'next', 'in particular', 'in sum', 'because',
                               'nevertheless', 'for instance', 'as a result', 'while', 'consequently', 'inasmuch as',
                               'in other words', 'thus', 'furthermore', 'yet', 'but', 'therefore', 'so', 'on the other hand',
                               'in addition', 'indeed', 'since', 'in fact', 'previously', 'subsequently', 'instead',
                               'by comparison', 'whereas', 'as', 'that is', 'moreover', 'by contrast', 'meanwhile',
                               'in the end', 'similarly', 'additionally', 'accordingly', 'at the time', 'even though',
                               'ultimately', 'still', 'likewise', 'for one thing', 'though', 'or', 'on the one hand', 'later',
                               'on the contrary', 'in turn', 'in contrast', 'earlier', 'further', 'besides', 'on the whole',
                               'first', 'overall', 'since then', 'simultaneously', 'when', 'soon', 'as it turns out', 'insofar as',
                               'hence', 'so that', 'ever since', 'finally', 'to this end', 'afterwards', 'in comparison', 'in summary',
                               'as a consequence', 'particularly', "what's more", 'after', 'thereafter', 'eventually', 'incidentally', 'before',
                               'for', 'plus', 'at the same time', 'separately', 'in the meantime', 'nonetheless', 'so far', 'in response', 'regardless',
                               'as a matter of fact', 'now', 'for one', 'second', 'third', 'in return', 'at that time']
            self.conn2i = dict((s,i)  for (i,s) in enumerate(self.conn) )




        else:
            self.sec_sense = [
                'Temporal.Asynchronous', 'Temporal.Synchronous', 'Contingency.Cause',
                'Contingency.Cause+Belief', 'Contingency.Condition', 'Contingency.Purpose',
                'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Equivalence',
                'Expansion.Level-of-detail', 'Expansion.Manner', 'Expansion.Substitution']
            self.sec_sense2i = dict((s,i) for i, s in enumerate(self.sec_sense))
            self.top2second = {'Temporal': 'Temporal.Asynchronous',
                               'Comparison': 'Comparison.Concession',
                               'Contingency': 'Contingency.Cause',
                               'Expansion': 'Expansion.Conjunction'}



        if args.split == 'ji':
            self.train_section = [i for i in range(2, 3)]
            self.dev_section = [i for i in range(0, 1)]
            self.test_section = [i for i in range(21, 22)]
            # self.train_section = [ i for i in range(2,21)]
            # self.dev_section = [i for i in range(0,2)]
            # self.test_section = [i for i in range(21,23)]
            self.section = {'train': self.train_section , 'dev':self.dev_section, 'test':self.test_section}
        if args.split == 'lin':
            self.train_section = [i for i in range(2, 22)]  # 2-21
            self.dev_section = [22] #
            self.test_section = [23] #
            self.section = {'train': self.train_section, 'dev': self.dev_section, 'test': self.test_section}

        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path, from_pt=True)
        self.dataset = self.read_data(args.pdtb_version, data_path, args.n_class, mode=self.mode, usecols=usecols)
        # self.compute_classnum(self.dataset, mode=self.mode)


    # 统计每个语义层次有多少类
    def compute_classnum(self, dataset, mode):
        if self.args.pdtb_version == 2:
            print('PDTB2.0, {}_dataset中, 计算该语义层次下 每个类别数目...'.format(mode))
            # class1fir, class1sec = defaultdict(int), defaultdict(int)
            # dataset['ConnHeadSemClass1']
            class1fir = dict(dataset.groupby('ConnHeadSemClass1Fir').size())
            class1sec = dict(dataset.groupby('ConnHeadSemClass1Sec').size())
            print('第一层级类别及个数')
            print(class1fir)
            print('第二层级类别及个数')
            print(class1sec)

        else:
            print('PDTB3.0, {}_dataset中, 计算该语义层次下 每个类别数目...'.format(mode))

            class1fir = dict(dataset.groupby('label').size()) if mode=='train' else dict(dataset.groupby('label1').size())
            class1sec = dict(dataset.groupby('ConnHeadSemClass1Sec').size())
            print('第一层级类别及个数')
            print(class1fir)
            print('第二层级类别及个数')
            print(class1sec)




    def get_class1sec(self, semclass):
        if len(semclass.split('.')[0:2]) > 1 and '.'.join(semclass.split('.')[0:2]) in self.sec_sense:
            return '.'.join(semclass.split('.')[0:2])
        else:
            return self.top2second[semclass.split('.')[0]]



    def read_data(self, pdtb_version, data_path, n_class, mode=None, usecols=None): # mode = ['train', 'dev', 'test']
        if pdtb_version == 2:
            dataset =  pd.read_csv(data_path, usecols=['Relation', 'Section', \
            'Conn1', 'ConnHeadSemClass1', 'Conn2', 'Conn2SemClass1', 'Arg1_RawText', 'Arg2_RawText'])
            # 筛选数据
            # secs = self.section[mode]
            dataset.query('Relation == \'Implicit\' and Section in @self.section[@mode]', inplace=True)

            if mode == 'train':
                newframe = dataset[pd.notnull(dataset["Conn2SemClass1"])]
                newframe['ConnHeadSemClass1'], newframe['Conn1'] = newframe['Conn2SemClass1'], newframe['Conn2']
                dataset = dataset.append(newframe)

            # 为dataset添加一列, 该列为论元1长度
            dataset['arg1_length'] = dataset.apply(lambda rows: len(rows['Arg1_RawText'].strip().split(' ')), axis=1)
            dataset['ConnHeadSemClass1Fir'] = dataset.apply(lambda rows: rows['ConnHeadSemClass1'].split('.')[0], axis=1)
            dataset['ConnHeadSemClass1Sec'] = dataset.apply(
                lambda rows: self.get_class1sec(rows['ConnHeadSemClass1']), axis=1)
            # data augument, 数据增强

            # dataset.query('ConnHeadSemClass1Sec in @self.sec_sense', inplace=True)

            # 暂时不对第三层语义进行处理
            # dataset['ConnHeadSemClass1Thi'] = dataset.apply(lambda rows: self.arg1_length(rows['Arg1_RawText']), axis=1)

            return dataset
        else:
            #对pdtb3.0进行处理，当文本文件中带有英文双引号时，直接用pd.read_csv进行读取会导致行数减少，此时应该对read_csv设置参数quoting=3
            dataset = pd.read_csv(data_path, delimiter='\t', usecols=usecols, quoting=3)
            dataset['arg1_length'] = dataset.apply(lambda rows: len(rows['arg1'].strip().split(' ')), axis=1)
            dataset['ConnHeadSemClass1Sec'] = dataset.apply(lambda rows:
                                                            self.get_class1sec(rows['full_sense']) if mode=='train'
                                                                                  else self.get_class1sec(rows['full_sense1']), axis=1)

            # dataset.query('ConnHeadSemClass1Sec in @self.sec_sense', inplace=True)


            # dataset.query('ConnHeadSemClass1Sec in @self.sec_sense', inplace=True)

            return dataset


    def __getitem__(self, item):
        if self.args.pdtb_version == 2:
            arg = CLS + self.dataset.iloc[item]['Arg1_RawText'] + MASK + self.dataset.iloc[item]['Arg2_RawText'] + SEP
            input = self.tokenizer(arg, truncation=True, max_length=self.args.max_length, padding='max_length')

            mask_idx = np.zeros(self.args.max_length)
            if self.dataset.iloc[item]['arg1_length'] + 1 < self.args.max_length:
                mask_idx[self.dataset.iloc[item]['arg1_length'] + 1] = 1
            else:
                mask_idx[0] = 1

            return \
                np.array(input['input_ids'], np.int32), np.array(input['attention_mask'], np.int32), \
                   np.array(self.fir_sense2i[self.dataset.iloc[item]['ConnHeadSemClass1Fir']], np.int32), \
                   np.array(self.sec_sense2i[self.dataset.iloc[item]['ConnHeadSemClass1Sec']], np.int32), \
                   np.array(self.conn2i[self.dataset.iloc[item]['Conn1']], np.int32), \
                   np.array(mask_idx, np.int32)

        if self.args.pdtb_version == 3:
            arg = CLS + self.dataset.iloc[item]['arg1'] + MASK + self.dataset.iloc[item]['arg2'] + SEP
            input = self.tokenizer(arg, truncation=True, max_length=self.args.max_length, padding='max_length')
            mask_idx = np.zeros(self.args.max_length)
            if self.dataset.iloc[item]['arg1_length'] + 1 < self.args.max_length:
                mask_idx[self.dataset.iloc[item]['arg1_length'] + 1] = 1
            else:
                mask_idx[0] = 1
            return np.array(input['input_ids'], np.int32), \
                   np.array(input['attention_mask'], np.int32), \
                   np.array(self.fir_sense2i[self.dataset.iloc[item]['label']], np.int32) if self.mode == 'train' else np.array(self.fir_sense2i[self.dataset.iloc[item]['label1']], np.int32), \
                   np.array(self.sec_sense2i[self.dataset.iloc[item]['ConnHeadSemClass1Sec']], np.int32),  \
                   np.array(mask_idx, np.int32)




    def __len__(self):
        return len(self.dataset)






    # def pad(self, batch):
    #     batch_size = len(batch)
    #     input_ids, attention_mask, firlabel, seclabel, connlabel, mask_idxs= [], [], [], [], [], []
    #
    #     for i in range(batch_size):
    #         input_ids.append(batch[i][0])
    #         attention_mask.append(batch[i][1])
    #         firlabel.append(batch[i][2])
    #         seclabel.append(batch[i][3])
    #         connlabel.append(batch[i][4])
    #         # labels2.append(batch[i][3])
    #         # mask_idx, label_idx = np.zeros(max_length), np.zeros(max_length)
    #         mask_idx = np.zeros(self.args.max_length)
    #
    #         # +1： 'CLS'
    #         if batch[i][4] + 1 < self.args.max_length:
    #             mask_idx[batch[i][4] + 1] = 1
    #         else:
    #             mask_idx[0] = 1
    #
    #
    #         mask_idxs.append(mask_idx.tolist())
    #
    #     input_ids = mindspore.tensor(input_ids).long()
    #     attention_mask = mindspore.tensor(attention_mask).long()
    #     firlabel = mindspore.tensor(firlabel).long()
    #     seclabel = mindspore.tensor(seclabel).long()
    #     connlabel = mindspore.tensor(connlabel).long()
    #     mask_idxs = mindspore.tensor(mask_idxs).long()
    #
    #
    #     return input_ids, attention_mask, firlabel, seclabel, connlabel, mask_idxs





