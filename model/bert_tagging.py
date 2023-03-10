import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.utils.rnn as rnn_utils
from model.slu_baseline_tagging import TaggingFNNDecoder


class BERTTagging(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        checkpoint = getattr(config, "ckpt", "bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.bert = BertModel.from_pretrained(checkpoint)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        embed_size = self.bert.config.to_dict()['hidden_size']
        self.tag_pad_idx = self.tokenizer.pad_token_id
        self.dense = nn.Linear(embed_size * 2, embed_size)
        if getattr(self.config, "use_rnn", False):
            self.rnn = getattr(nn, config.encoder_cell)(embed_size,
                                                        config.hidden_size // 2,
                                                        num_layers=config.num_layer,
                                                        bidirectional=True,
                                                        batch_first=True)
            embed_size = config.hidden_size
        self.output_layer = TaggingFNNDecoder(embed_size, config.num_tags, self.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = []
        input_ids = []
        for text in batch.utt:
            tokens = self.tokenizer.encode_plus(text,
                                                add_special_tokens=False,
                                                max_length=batch.max_len,
                                                pad_to_max_length=True,
                                                return_attention_mask=True)
            mask = tokens["attention_mask"]
            ids = tokens["input_ids"]
            tag_mask.append(mask)
            input_ids.append(ids)
        tag_mask = torch.tensor(tag_mask, dtype=torch.float, device="cuda")
        input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        bert_out = self.bert(input_ids=input_ids, attention_mask=tag_mask, output_hidden_states=True)
        hiddens = bert_out.last_hidden_state

        if getattr(self.config, "bert_as_embed", False):
            hiddens = hiddens.detach()
        if getattr(self.config, "use_rnn", False):
            packed_inputs = rnn_utils.pack_padded_sequence(hiddens,
                                                           batch.lengths,
                                                           batch_first=True,
                                                           enforce_sorted=True)
            packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
            hiddens, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
            hiddens = self.dropout_layer(hiddens)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)
        return tag_output

    def _inference(self, batch):
        input_ids = []
        attention_mask = []
        for text in batch.utt:
            tokens = self.tokenizer.encode_plus(text,
                                                add_special_tokens=False,
                                                truncation=True,
                                                max_length=batch.max_len,
                                                padding='max_length',
                                                return_attention_mask=True)
            mask = tokens["attention_mask"]
            ids = tokens["input_ids"]
            attention_mask.append(mask)
            input_ids.append(ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float, device=self.device)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hiddens = self.dropout_layer(bert_out.last_hidden_state)
        tag_output = self.output_layer(hiddens, attention_mask)
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)

        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()

    def inference(self, label_vocab, batch):
        batch_size = len(batch)
        prob = self._inference(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions