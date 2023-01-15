import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import EmbeddingUtils
from utils.evaluator import Evaluator
import jieba


class Example():

    @classmethod
    def configuration(cls, root, train_path=None, embedding_path=None, segmentation=False):
        cls.evaluator = Evaluator()
        cls.segmentation = segmentation
        if not segmentation:
            cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        else:
            cls.word_vocab = Vocab(padding=True, unk=True)
            cls.word_vocab.from_train_segmentation(filepath=train_path)
        cls.embedding = EmbeddingUtils(embedding_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex
        if not self.segmentation:
            self.utt = self.preprocessing(ex['manual_transcript'])
            self.slot = {}
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
            self.tags = ['O'] * len(self.utt)
            for slot in self.slot:
                value = self.slot[slot]
                bidx = self.utt.find(value)
                if bidx != -1:
                    self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                    self.tags[bidx] = f'B-{slot}'
            self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
            self.input_idx = [Example.word_vocab[c] for c in self.utt]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        else:
            utt_uncut = self.preprocessing(ex['manual_transcript'])
            self.utt = jieba.lcut(utt_uncut,cut_all=False)
            idx = []
            for word in self.utt:
                idx.append(utt_uncut.find(word))

            self.slot = {}
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    value = label[2]
                    value_idx = utt_uncut.find(value)
                    value_list = []
                    for i in range(len(idx)):
                        if value_idx <= idx[i] and value_idx + len(value) > idx[i]:
                            value_list.append(self.utt[i])
                    self.slot[act_slot] = value_list
            self.tags = ['O'] * len(self.utt)
            for slot in self.slot:
                value = self.slot.get(slot, None)
                if not value:
                    continue
                if value[0] in self.utt:
                    bidx = self.utt.index(value[0])
                else:
                    bidx = -1
                if bidx != -1:
                    self.tags[bidx:bidx + len(value)] = [f'I-{slot}'] * len(value)
                    self.tags[bidx] = f'B-{slot}'
            self.slotvalue = [f'{slot}-{"".join(value)}' for slot, value in self.slot.items()]
            self.input_idx = [Example.word_vocab[c] for c in self.utt]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]

    def preprocessing(self, utt):
        if utt.find('(')!= -1:
            utt.replace("(unknown)","")
            utt.replace("(side)","")
            utt.replace("(robot)","")
            utt.replace("(noise)","")
            utt.replace("(dialect)","")
        return utt
        
class DevExample():

    @classmethod
    def configuration(cls, segmentation=False):
        cls.segmentation = segmentation

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        examples = sorted(examples, key=lambda x: len(x.input_idx), reverse=True)
        return examples

    def __init__(self, ex: dict):
        super(DevExample, self).__init__()
        self.ex = ex
        if not self.segmentation:
            self.utt = ex['asr_1best']
            self.slot = {}
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
            self.tags = ['O'] * len(self.utt)
            for slot in self.slot:
                value = self.slot[slot]
                bidx = self.utt.find(value)
                if bidx != -1:
                    self.tags[bidx:bidx + len(value)] = [f'I-{slot}'] * len(value)
                    self.tags[bidx] = f'B-{slot}'
            self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
            self.input_idx = [Example.word_vocab[c] for c in self.utt]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        else:
            utt_uncut = ex['asr_1best']
            self.utt = jieba.lcut(utt_uncut, cut_all=False)
            idx = []
            for word in self.utt:
                idx.append(utt_uncut.find(word))

            self.slot = {}
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    value = label[2]
                    value_idx = utt_uncut.find(value)
                    value_list = []
                    for i in range(len(idx)):
                        if value_idx <= idx[i] and value_idx + len(value) > idx[i]:
                            value_list.append(self.utt[i])
                    self.slot[act_slot] = value_list
            self.tags = ['O'] * len(self.utt)
            for slot in self.slot:
                value = self.slot.get(slot, None)
                if not value:
                    continue
                if value[0] in self.utt:
                    bidx = self.utt.index(value[0])
                else:
                    bidx = -1
                if bidx != -1:
                    self.tags[bidx:bidx + len(value)] = [f'I-{slot}'] * len(value)
                    self.tags[bidx] = f'B-{slot}'
            self.slotvalue = [f'{slot}-{"".join(value)}' for slot, value in self.slot.items()]
            self.input_idx = [Example.word_vocab[c] for c in self.utt]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]


class TestExample():

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        examples = sorted(examples, key=lambda x: len(x.input_idx), reverse=True)
        return examples

    def __init__(self, ex: dict):
        super(TestExample, self).__init__()
        self.ex = ex
        if not Example.segmentation:
            self.utt = ex['asr_1best']
            self.tags = ['O'] * len(self.utt)
            self.input_idx = [Example.word_vocab[c] for c in self.utt]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        else:
            self.utt_uncut = ex['asr_1best']
            self.utt = jieba.lcut(self.utt_uncut, cut_all=False)
            self.tags = ['O'] * len(self.utt)
            self.input_idx = [Example.word_vocab[c] for c in self.utt]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
