
import torch

import mindspore
from mindspore.experimental.optim import AdamW

from mindspore.experimental.optim.lr_scheduler import LRScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score


import numpy as np

from evaluation import compute_acc_f1
import warnings

warnings.filterwarnings("ignore")

class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,  args, patience=10, verbose=True, delta=0,):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.macro_f1_min = np.Inf
        self.delta = delta
        self.args = args

    def __call__(self, macro_f1, model, optimizer):
        score = macro_f1
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            # if self.counter==1:
            #     torch.save(attention_weights, 'attention_weights')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, optimizer)
            self.counter = 0


    def save_checkpoint(self, model, optimizer):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation F1 increased ({self.macro_f1_min:.6f} --> {self.best_score:.6f}).  Saving model ...')
            saved_path = './saved_model/PDTB'+str(self.args.pdtb_version)+'/'+str(self.args.n_class)+'/'
            # torch.save(model.state_dict(), saved_path+'model.pt')  # 这里会存储迄今最优的模型
            # torch.save(optimizer.state_dict(), saved_path+'opt.pt')

        self.macro_f1_min = self.best_score



def label_trans(firlabel, seclabel, config):
    """
    :@param:
    @return: matrixed_label: 转化为层次化形式, pdtb2.0: [batch, 4+11]
    """
    batch_size = firlabel.shape(0)
    for i in range(batch_size):
        hier_label = mindspore.ops.zeros(config.fir_num + config.sec_num)
        hier_label[firlabel] = 1
        hier_label[config.fir_num + seclabel] = 1
        if i == 0:
            hier_labels = hier_label.unsqueeze(dim=0)
        else:
            hier_labels = mindspore.ops.cat((hier_labels, hier_label.unsqueeze(0)), axis=0)
    return hier_labels

def train_one_epoch(data, model, logger, args, config, k, **kwargs):
    best_test_acc = 0.0
    device = args.device
    train_data = data.train_dl
    dev_data = data.dev_dl
    test_data = data.test_dl
    # model = model.to(device)

    # es = EarlyStopping(args, patience=5)

    cross_loss = kwargs.get('cross_loss', None)
    euc_loss = kwargs.get('euc_loss', None)
    con_loss = kwargs.get('con_loss', None)
    instance_loss = kwargs.get('instance_loss', None)
    label_loss = kwargs.get('label_loss', None)
    LabelRegularizer = kwargs.get('LabelRegularizer', None)

    lamb_cross = kwargs.get('lamb_cross', 1)
    lamb_euc = kwargs.get('lamb_euc', 0)
    lamb_con = kwargs.get('lamb_con', 0)
    lamb_instance = kwargs.get('lamb_instance', 0)
    lamb_label = kwargs.get('lamb_label', 0)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    t_total = train_data.get_dataset_size() // train_batch_size * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay),
    # # 分层学习率设置: BERT中对预训练模型设置较小学习率, 保持参数较好水平; 对下游设置较大学习率, 其中，n是模型每层名字，p是每层参数，model是模型名字。
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for p in model.trainable_params() if not any(nd in p.name for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for p in model.trainable_params() if any(nd in p.name for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    optimizer = AdamW(model.trainable_params() , lr=args.learning_rate, eps=args.adam_epsilon)

    def forward_fn(inputs, labels):

        input_ids, attention_mask, mask_idxs = inputs
        firlabel, seclabel, connlabel = labels
        # hier_labels = label_trans(firlabel, seclabel, config)
        result = model(input_ids=input_ids, attention_mask=attention_mask,mask_idxs=mask_idxs)
        tr_batch_loss = lamb_cross * (cross_loss(result['fir_logits'], firlabel) +
                                      cross_loss(result['sec_logits'], seclabel) +
                                      cross_loss(result['conn_logits'], connlabel))

        a = lamb_instance * instance_loss(result['mask_repr'], result['label_embeds'], connlabel)

        b = lamb_label * label_loss(result['label_embeds'], firlabel, seclabel, connlabel)

        tr_batch_loss = tr_batch_loss + a + b


        return tr_batch_loss, result['fir_logits'], result['sec_logits'], result['conn_logits']

    grad_fn = mindspore.value_and_grad(forward_fn, None, model.trainable_params(),  has_aux=True)

    def train_step(inputs, labels):
        (tr_batch_loss, fir_logits, sec_logits, conn_logits),  grads = grad_fn(inputs, labels)
        optimizer(grads)
        return tr_batch_loss, fir_logits, sec_logits, conn_logits

    for epoch in range(args.num_train_epochs):
        logger.info('***** epoch:{}, start training *****'.format(epoch))
        # tr_batch_loss, tr_loss = 0.0, 0.0
        tr_loss = 0
        # train_f1, f1= 0.0, 0.0
        fir_level_acc, sec_level_acc, thi_level_acc = 0.0, 0.0, 0.0
        fir_level_f1, sec_level_f1, thi_level_f1 = 0.0, 0.0, 0.0
        model.set_train()
        fir_preds, fir_truth, sec_preds, sec_truth, conn_preds, conn_truth = None, None, None, None, None, None
        for i, data in enumerate(train_data):
            input_ids, attention_mask, firlabel, seclabel, connlabel, mask_idxs = data
            inputs = (input_ids,attention_mask,mask_idxs)
            labels = (firlabel, seclabel, connlabel)
            tr_batch_loss, fir_logits, sec_logits, conn_logits = train_step(inputs, labels)
            tr_loss +=  tr_batch_loss

            if fir_preds is None :
                fir_preds = fir_logits.asnumpy()
                fir_truth = data[2].asnumpy()

            else:
                fir_preds = np.append(fir_preds, fir_logits.asnumpy(), axis=0)
                fir_truth = np.append(fir_truth, data[2].asnumpy(), axis=0)


            if sec_preds is None:
                sec_preds =  sec_logits.asnumpy()
                sec_truth = data[3].asnumpy()
            else:
                sec_preds = np.append(sec_preds, sec_logits.asnumpy(), axis=0)
                sec_truth = np.append(sec_truth, data[3].asnumpy(), axis=0)
            if conn_preds is None:
                conn_preds = conn_logits.asnumpy()
                conn_truth = data[4].asnumpy()
            else:
                conn_preds = np.append(conn_preds, conn_logits.asnumpy(), axis=0)
                conn_truth = np.append(conn_truth, data[4].asnumpy(), axis=0)



            if i >= 0 and i % args.print_batch_num == 0:
                if i == 0:
                    # logger.info('        loss |    Temporal | Contingency |  Comparison |   Expansion |   micro_acc |    macro_f1 ')
                    logger.info('        loss |   fir_acc |    fir_macro_f1 | sec_acc | sec_marco_f1 | conn_acc | conn_marco_f1 ')

                else:

                    fir_acc, fir_macro_f1, sec_acc, sec_marco_f1, conn_acc, conn_marco_f1 = compute_acc_f1(np.argmax(fir_preds, axis=1), fir_truth, np.argmax(sec_preds, axis=1), sec_truth, np.argmax(conn_preds, axis=1), conn_truth, args, config)
                    logger.info('{:13.5f}|{:13.5f}|{:13.5f}|{:13.5f}|{:13.5f}|{:13.5f}|{:13.5f}'.format(tr_loss.item()/i, fir_acc, fir_macro_f1, sec_acc, sec_marco_f1, conn_acc, conn_marco_f1))

        logger.info('***** epoch:{}, begin deving *****'.format(epoch))
        test(dev_data, model, logger, args, config, **kwargs)

        logger.info('***** epoch:{}, begin testing *****'.format(epoch))
        test_res = test(test_data, model, logger, args, config, test=True, **kwargs)

        if test_res['fir_macro_f1'] > best_test_acc:
            best_test_acc = test_res['fir_macro_f1']


        # es(test_res['fir_macro_f1'], model, optimizer)
        # if es.early_stop == True:
        #     print('Early Stopping ! the best macro_f1 : {}'.format(es.best_score))
        #     break

    return

def test(test_data, model, logger, args, config, test=False, cm=False, **kwargs):

    cross_loss = kwargs.get('cross_loss', None)
    euc_loss = kwargs.get('euc_loss', None)
    con_loss = kwargs.get('con_loss', None)
    instance_loss = kwargs.get('instance_loss', None)
    label_loss = kwargs.get('label_loss', None)
    LabelRegularizer = kwargs.get('LabelRegularizer', None)
    lamb_cross = kwargs.get('lamb_cross', 1)


    lamb_euc = kwargs.get('lamb_euc', 0)
    lamb_con = kwargs.get('lamb_con', 0)
    lamb_instance = kwargs.get('lamb_instance', 0)
    lamb_label = kwargs.get('lamb_label', 0)


    best_acc, best_f1 = 0.0, 0.0
    device = args.device
    size = len(test_data)
    fir_acc, sec_acc, thi_acc = 0.0, 0.0, 0.0
    fir_macro_f1 = 0.0
    fir_f1, sec_f1, thi_f1 = 0.0, 0.0, 0.0
    best_fir_f1, best_sec_f1, best_thi_f1 = 0.0, 0.0, 0.0
    te_loss = 0.0
    fir_preds, sec_preds, conn_preds, fir_truth, sec_truth, conn_truth = None, None, None, None, None, None

    model.set_train(False)

    for i, data in enumerate(test_data):
        input_ids, attention_mask, firlabel, seclabel, connlabel, mask_idxs = data
        inputs = (input_ids, attention_mask, mask_idxs)
        labels = (firlabel, seclabel, connlabel)

        result = model(input_ids=input_ids, attention_mask=attention_mask,mask_idxs=mask_idxs)
        test_loss = lamb_cross * (cross_loss(result['fir_logits'], firlabel) +
                                      cross_loss(result['sec_logits'], seclabel) +
                                      cross_loss(result['conn_logits'], connlabel))

        a = lamb_instance * instance_loss(result['mask_repr'], result['label_embeds'], connlabel)

        b = lamb_label * label_loss(result['label_embeds'], firlabel, seclabel, connlabel)


        if instance_loss is not None:
            test_loss += lamb_instance * instance_loss(result['mask_repr'], result['label_embeds'], connlabel)
        if label_loss is not None:
            test_loss += lamb_label * label_loss(result['label_embeds'], firlabel, seclabel, connlabel)

        te_loss += test_loss.item()

        if fir_preds is None:
            fir_preds = result['fir_logits'].asnumpy()
            # preds = np.append(preds, result['first_level_contrast_logits'].data().cpu().numpy(), axis=0)
            fir_truth = firlabel.asnumpy()
            # truth = np.append(truth, fir_label1.data().cpu().numpy(), axis=0)
        else:
            fir_preds = np.append(fir_preds, result['fir_logits'].asnumpy(), axis=0)

            fir_truth = np.append(fir_truth, firlabel.asnumpy(), axis=0)

        if sec_preds is None:
            sec_preds = result['sec_logits'].asnumpy()
            sec_truth = seclabel.asnumpy()
        else:
            sec_preds = np.append(sec_preds, result['sec_logits'].asnumpy(), axis=0)
            sec_truth = np.append(sec_truth, seclabel.asnumpy(), axis=0)
        if conn_preds is None:
            conn_preds = result['conn_logits'].asnumpy()
            conn_truth = connlabel.asnumpy()
        else:
            conn_preds = np.append(conn_preds, result['conn_logits'].asnumpy(), axis=0)
            conn_truth = np.append(conn_truth, connlabel.asnumpy(), axis=0)


    logger.info(' test_loss |   fir_acc |    fir_macro_f1 | sec_acc | sec_marco_f1 | conn_acc | conn_marco_f1 ')

    fir_acc, fir_macro_f1, sec_acc, sec_marco_f1, conn_acc, conn_marco_f1 = compute_acc_f1(np.argmax(fir_preds, axis=1),
                                                                                           fir_truth,
                                                                                           np.argmax(sec_preds, axis=1),
                                                                                           sec_truth,
                                                                                           np.argmax(conn_preds,
                                                                                                     axis=1),
                                                                                           conn_truth, args, config)
    logger.info('{:13.5f}|{:13.5f}|{:13.5f}|{:13.5f}|{:13.5f}|{:13.5f}|{:13.5f}'.format(te_loss/ size, fir_acc,
                                                                                        fir_macro_f1, sec_acc,
                                                                                        sec_marco_f1, conn_acc,
                                                                                        conn_marco_f1))


    if not test:
        return
    else:
        return {
            'fir_acc':fir_acc,
            'fir_macro_f1':fir_macro_f1,
            'sec_acc':sec_acc,

        }





