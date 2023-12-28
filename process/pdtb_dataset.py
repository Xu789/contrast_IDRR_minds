
from process.base_dataset import BasePdtbDataset


from mindspore.dataset import GeneratorDataset


class Pdtb2Dataset:
    def __init__(self, args):
        # super(Pdtb2Dataset, self).__init__(args)
        mode = ['train', 'dev', 'test']
        column_names = ["input_ids","attention_mask","firlabel", "seclabel", "connlabel", "mask_idxs"]

        traind, devd, testd = GeneratorDataset(BasePdtbDataset(args, mode=mode[0]), column_names=column_names),\
            GeneratorDataset(BasePdtbDataset(args, mode=mode[1]), column_names=column_names), \
            GeneratorDataset(BasePdtbDataset(args, mode=mode[2]), column_names=column_names)

        # train_dataloader

        kwargs = {'batch_size': args.per_gpu_train_batch_size * args.n_gpu, 'shuffle': args.shuffle, 'drop_last': args.drop_last}
        self.train_dl = traind.batch(kwargs['batch_size'])
        self.dev_dl = devd.batch(kwargs['batch_size'])
        self.test_dl = testd.batch(kwargs['batch_size'])



class Pdtb3Dataset:
    def __init__(self, args):
        # super(Pdtb3Dataset, self).__init__(args)

        mode = ['train', 'dev', 'test']
        tr_usecols = ['label', 'arg1', 'arg2', 'conn', 'full_sense']
        te_usecols = ['label1', 'arg1', 'arg2', 'conn1', 'full_sense1']

        train_kwargs = {'data_path': args.data_path+mode[0]+'.tsv', 'usecols':tr_usecols}
        dev_kwargs = {'data_path': args.data_path+mode[1]+'.tsv', 'mode':'dev', 'usecols':te_usecols}
        test_kwargs = {'data_path': args.data_path+mode[2]+'.tsv', 'mode': 'test', 'usecols': te_usecols}
        traind, devd, testd = \
            BasePdtbDataset(args, **train_kwargs), \
            BasePdtbDataset(args, **dev_kwargs), \
            BasePdtbDataset(args, **test_kwargs)

        kwargs = {'batch_size': args.per_gpu_train_batch_size * args.n_gpu, 'shuffle': args.shuffle, 'drop_last': args.drop_last}

        # self.train_dl = DataLoader(traind, **kwargs, collate_fn=traind.pad)
        # self.dev_dl = DataLoader(devd, **kwargs, collate_fn=devd.pad)
        # self.test_dl = DataLoader(testd, **kwargs, collate_fn=testd.pad)


'''for test'''
if __name__ == '__main__':
    import argparse
    def parse():
        parser = argparse.ArgumentParser(description='')

        parser.add_argument('-sp', '--split', type=str, default='ji',
                            help='data splitting method')

        parser.add_argument('-dp', '--data_path', type=str, default='../corpus/pdtb3_ji/',
                            help='data path')
        # 2.0:data_path = '../corpus/pdtb2.csv'; 3.0:data_path = '../corpus/pdtb3_ji/'

        parser.add_argument('-pv', '--pdtb_version', type=int, default=3, choices=[2, 3],
                            help='the version for PDTB corpus')
        parser.add_argument('-m', '--model', type=str, default='roberta-base',
                            help='the base model ')
        parser.add_argument('-mp', '--model_path', type=str, default='../roberta-base',
                            help='the path for base model ')

        parser.add_argument('-nc', '--n_class', type=int, default=4,
                            help='the class for n ')
        parser.add_argument('-ml', '--max_length', type=int, default=100,
                            help='the max length for arg1+arg2')


        parser.add_argument('-pgtbs', '--per_gpu_train_batch_size', type=int, default=128,
                            help='the batch size for per gpu device in the training..')
        parser.add_argument('-ng', '--n_gpu', type=int, default=1,
                            help='choose the number of cuda')
        parser.add_argument('--shuffle', type=str, default=True,
                            help='whether to shuffle the data')
        parser.add_argument('-drl', '--drop_last', type=str, default=False,
                            help='whether to drop the last data that is less than one batch')
        return parser

    args = parse().parse_args()
    dataset = Pdtb2Dataset(args)
    train_dl = dataset.train_dl
    dev_dl = dataset.dev_dl
    test_dl = dataset.test_dl

    for i, x in enumerate(train_dl):
        # print(x.size())
        print(x)

    print('结束！')









