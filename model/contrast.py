
import mindspore.nn as nn
import mindspore
from mindspore.ops import Dropout
from mindnlp.transformers import RobertaTokenizer, RobertaModel

import numpy as np
# from mindformers.models.bert import BertForPreTraining, BertConfig, BertModel, BertTokenizer
from mindspore import ms_function


class ContrastModel(nn.Cell):

    def __init__(self, args, config, **kwargs):
        super().__init__()
        self.config = config
        self.args = args
        self.threshold = args.threshold
        self.tau = args.tau
        self.num_gcn_layer = args.num_gcn_layer

        self.cross_loss = kwargs.get('cross_loss', None)
        self.euc_loss = kwargs.get('euc_loss', None)
        self.con_loss = kwargs.get('con_loss', None)
        self.instance_loss = kwargs.get('instance_loss', None)
        self.label_loss = kwargs.get('label_loss', None)

        self.lamb_cross = kwargs.get('lamb_cross', 1)
        self.lamb_euc = kwargs.get('lamb_euc', 0)
        self.lamb_con = kwargs.get('lamb_con', 0)

        self.model = RobertaModel.from_pretrained(args.model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path,from_pt=True)

        # bert_config = BertConfig.from_pretrained(args.model_path)
        self.embedding = RobertaModel.from_pretrained(args.model_path).embeddings
        # self.embedding.requires_grad = False
        self.dropout = Dropout(0.07)

        label_num = config.label_num
        embed_size = config.label_embedding_size
        self.n_class = args.n_class
        self.num_gcn_layer = args.num_gcn_layer
        self.label_name = self.config.label

        self.ec_label= self.tokenizer(self.label_name, padding='max_length', truncation=True)
        #  self.ec_label: self.encoded_labels: [117, 10]
        print(type(mindspore.tensor(self.ec_label['input_ids'])))
        # self.label_embeds = self.embedding(mindspore.tensor(self.ec_label['input_ids'])).sum(dim=1)
        # self.label_embeds = self.label_embeds.sum(dim=1)
        # self.label_embeds = mindspore.Parameter(self.embedding(self.label_embeds , requires_grad=True))
        # self.label_embeds = mindspore.nn.Embedding(117, 768)
        # x = mindspore.Tensor(np.array([i for i in range(117)]), mindspore.int32)




        self.fir_classifier = nn.Dense(768, 4)
        if args.pdtb_version == 2:
            self.sec_classifier = nn.Dense(768, 11)
            self.conn_classifier = nn.Dense(768, 102)
        else:
            self.sec_classifier = nn.Dense(768, 14)
            self.conn_classifier = nn.Dense(768, 181)

        self.softmax = nn.Softmax(axis=-1)

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        embedding_weight=None,
        mask_idxs=None,
        label_idxs=None,
        args=None,
        weight=None,
        training=True,
    ):
        """
        @param label: [batch]
        @param mask_idxs: [batch, seq_length]
        @param label_indes: [batch, seq_length]
        @return:
        """

        # loss = 0.0
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        contrast_mask = None
        label_embeds = self.embedding(mindspore.tensor(self.ec_label['input_ids']))
        label_embeds = label_embeds.sum(axis=1)

        # self.bert_model =  ms_function(fn=self.model)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask)


        # pooled_output = self.dropout(self.pooler(outputs[0]))
        # outputs[0]为last_hidden_state, [batch, seq_len, 768]
        # self.pooler类获取序列第一个词[CLS]的隐藏层表示
        pooled_output = outputs[0]
        mask_repr = pooled_output[mask_idxs == 1]
        sen_repr = (outputs[0].sum(dim=1) / outputs[0].size(1) - mask_repr) * 0.1 + mask_repr * 0.9
        sen_repr = self.dropout(sen_repr)

        fir_logits = self.fir_classifier(sen_repr)
        sec_logits = self.sec_classifier(sen_repr)
        conn_logits = self.conn_classifier(sen_repr)

        return {
            'fir_logits': fir_logits,
            'sec_logits': sec_logits,
            'conn_logits': conn_logits,
            'mask_repr': mask_repr,
            'sen_repr': sen_repr,
            'label_embeds': label_embeds,
        }
