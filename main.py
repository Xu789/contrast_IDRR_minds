# coding: UTF-8

import argparse
import logging

import mindspore


from mindspore.nn  import CrossEntropyLoss


from SupConLoss import  Instance2label, Label2label
from config import Config
from model.contrast import ContrastModel
from process.pdtb_dataset import Pdtb2Dataset, Pdtb3Dataset
from mindspore import context

from train import train_one_epoch, test




def log():
    logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def parse():
    parser = argparse.ArgumentParser(description='Contrast Learning for Implicit Discourse Relation Recognition')

    parser.add_argument('--cuda_id', default=0, type=int,
                        help='cuda id')
    parser.add_argument('-sp', '--split', type=str, default='ji',
                        help='data splitting method')
    parser.add_argument('-dp', '--data_path', type=str, default='./corpus/pdtb2.csv',
                        help='data path')
    # 2.0:data_path = './corpus/pdtb2.csv'; 3.0:data_path = './corpus/pdtb3_ji/'
    parser.add_argument('-pv', '--pdtb_version', type=int, default=2, choices=[2, 3],
                        help='the version for PDTB corpus')
    parser.add_argument('-m', '--model', type=str, default='roberta-base',
                        help='the base model ')
    parser.add_argument('-mp', '--model_path', type=str, default='roberta-base',
                        help='the path for base model ')

    parser.add_argument('-nc', '--n_class', type=int, default=4,
                        help='the class for n ')
    parser.add_argument('-ml', '--max_length', type=int, default=100,
                        help='the max length for arg1+arg2')

    parser.add_argument('-pgtbs', '--per_gpu_train_batch_size', type=int, default=8,
                        help='the batch size for per gpu device in the training..')
    parser.add_argument('-ng', '--n_gpu', type=int, default=1,
                        help='choose the number of cuda')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='whether to shuffle the data')
    parser.add_argument('-drl', '--drop_last', type=bool, default=False,
                        help='whether to dro p the last data that is less than one batch')

    parser.add_argument('-lgp', '--label_graph_path', type=str, default='./label_graph.g',
                        help='the path of label_graph')


    parser.add_argument('-de', '--device', type=str, default='cuda', help='the device (cuda/cpu) for learning...')


    # parser.add_argument('-nc', '--num_class', type=int, default=117,
    #                     help='the number of multi-labels')

    parser.add_argument('--seed', type=int, default=40,
                        help='the random seed for initialization')
    parser.add_argument('-nte', '--num_train_epochs', type=int, default=100,
                        help='the num epoches for training...')
    parser.add_argument('-gas', '--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="weight_decay.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="weight_decay.") # 5e-6
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="weight_decay.")
    parser.add_argument("--warm_up_ratio", default=0.6, type=float,
                        help="warm_up_ratio.")

    # parser.add_argument('--lamb', default=1, type=float,
    #                     help='lambda')
    parser.add_argument('--contrast', default=1, type=int,
                        help='Whether use contrastive model.')
    # parser.add_argument('--graph', default=1, type=int,
    #                     help='Whether use graph encoder.')
    # parser.add_argument('--layer', default=1, type=int,
    #                     help='Layer of Graphormer.')
    parser.add_argument('--max_grad_norm', default=0.1, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--threshold', default=0.01, type=float,
                        help='Threshold for keeping tokens. Denote as gamma in the paper.')
    parser.add_argument('--tau', default=1, type=float,
                        help='Temperature for contrastive model.')
    # parser.add_argument('--warmup', default=2000, type=int,
    #                     help='the number of iterations for the increasing lr')

    parser.add_argument('-rsn', '--random_start_num', default=5, type=int,
                        help='the number of random start...')
    parser.add_argument('--num_gcn_layer', default=2, type=int,
                        help='the num of gcn')
    parser.add_argument('--lamb_cross', default=1, type=float,
                        help='the lambda for the weight of contrast loss.' )
    parser.add_argument('--label_smooth', default=0, type=int,
                        help='if replace cross entroy with label_smooth, default : no replace.' )

    parser.add_argument('--lamb_con', default=1.0, type=float,
                        help='the lambda for the weight of contrast loss.' )
    parser.add_argument('--lamb_euc', default=0.1, type=float,
                        help='the lambda for the weight of contrast loss.' )
    parser.add_argument('--lamb_instance', default=1.0, type=float,
                        help='the lambda for the weight of InstaceAnchor loss.' )
    parser.add_argument('--lamb_label', default=1.0, type=float,
                        help='the lambda for the weight of LabelAnchor loss.' )


    parser.add_argument('--temperature', default=0.5, type=float,
                        help='the lambda for the weight of contrast loss.' )

    parser.add_argument('--print_batch_num', default=50, type=int,
                        help='the num for print acc/f1 information' )


    return parser




def main():
    args = parse().parse_args()

    context.set_context(device_target="GPU")
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
    logger = log()
    logger.info('args: {}'.format(args.__dict__))


    # 获取数据
    data = Pdtb2Dataset(args) if args.pdtb_version == 2 else Pdtb3Dataset(args)
    for i, batch in enumerate(data.train_dl):
        if i<3:
            print(len(batch))
    config = Config(args)

    kwargs = {'cross_loss': CrossEntropyLoss(),
            # 'euc_loss': nn.PairwiseDistance(p=2.0),  # Euclidean
            # 'con_loss': SupConLossSec(temp=args.temperature),
            # 'con_loss': SupConLoss(args.n_class, args.per_gpu_train_batch_size, w_13=1.5, w_23=1),
            'instance_loss': Instance2label(args, args.temperature),
            'label_loss': Label2label(args, args.temperature),
            'LabelRegularizer':None,
            'lamb_cross':args.lamb_cross,
            'lamb_euc':args.lamb_euc,
            'lamb_con':args.lamb_con,
            'lamb_instance':args.lamb_instance,
            'lamb_label': args.lamb_label,
            }

    for k in range(args.random_start_num): # rsn:random_start_num
        model = ContrastModel(args, config, **kwargs)
        train_one_epoch(data, model, logger, args,  config, k, **kwargs)


if __name__ == '__main__':
    main()
    # import mindspore
    # from mindnlp.transformers import RobertaTokenizer, RobertaModel
    #
    # embedding = RobertaModel.from_pretrained('roberta-base').embeddings
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # input_ids = mindspore.tensor(mindspore.numpy.rand((117, 512)))
