import argparse


def parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-sp', '--split', type=str, default='ji',
                        help='data splitting method')

    parser.add_argument('-dp', '--data_path', type=str, default='../corpus/pdtb2.csv',
                        help='data path')
    parser.add_argument('-pv', '--pdtb_version', type=int, default=2, choices=[2, 3],
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






