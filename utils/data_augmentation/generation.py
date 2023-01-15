import json
import random
import torch
import torch.nn as nn
from transformers import EncoderDecoderModel, BertTokenizer
import sys,os
sys.path.append('.')
# from utils.data_augmentation.train_aug import Augmentor
# from utils.data_augmentation.augmentation import Delexical
from nltk.metrics import edit_distance

class Augmentor(nn.Module):
    def __init__(self, ckpt="hfl/chinese-bert-wwm-ext"):
        super().__init__()
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(ckpt, ckpt)
        self.tokenizer = BertTokenizer.from_pretrained(ckpt)
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def forward(self, src_dataset, tgt_dataset):
        input_ids = self.tokenizer(src_dataset, add_special_tokens=False, return_tensors="pt", padding="longest").input_ids
        input_ids = input_ids.cuda()
        labels = self.tokenizer(tgt_dataset, return_tensors="pt", padding="longest").input_ids
        labels = labels.to("cuda")
        loss = self.model(input_ids=input_ids, labels=labels).loss
        return loss

    def generate(self, src_dataset):
        input_ids = self.tokenizer(src_dataset, add_special_tokens=False, return_tensors="pt", padding="longest").input_ids
        input_ids = input_ids.to("cuda")
        generated_ids = self.model.generate(input_ids, max_length=50)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for i in range(len(generated_text)):
            generated_text[i] = generated_text[i].replace(" ", "")
        return generated_text

class Delexical():
    @classmethod
    def load_dataset(cls, data_path="./data/train_original.json"):
        datas = json.load(open(data_path, 'r', encoding="utf-8"))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                if ex.utt == "":
                    continue
                if "<" not in ex.utt and ">" not in ex.utt: 
                    continue
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Delexical, self).__init__()
        self.utt = ex['manual_transcript']
        self.ori = ex['manual_transcript']
        for i in ["(unknown)", "(robot)", "(dialect)", "(side)"]:
            self.utt = self.utt.replace(i, "")
        self.slot = []
        for label in ex['semantic']:
            if label[0] == "inform":
                act = "通知"
            else:
                act = "否定"
            act_slot = f'<{act}-{label[1]}>'
            if len(label) == 3:
                self.slot.append((act_slot, label[2]))

        for sem in self.slot:
            act_slot = sem[0]
            value = sem[1]
            idx = self.utt.find(value)
            if idx == -1:
                continue
            tmp = list(self.utt)
            tmp[idx:idx+len(value)] = act_slot
            self.utt = "".join(tmp)
        self.slot = tuple(sorted(map(lambda i: i[0], self.slot)))


batch = 256
num_ranks = 5
random.seed(999)
model = Augmentor()
model.to("cuda")
checkpoint = torch.load("model_aug.bin")
model.load_state_dict(checkpoint["model"])
print("Model loaded successfully")


delexical_ex = Delexical.load_dataset()
src_dataset = []
for ex in delexical_ex:
    for i in range(num_ranks):
        src_dataset.append(ex.utt+f"#{i+1}")
print(f"Dataset loaded successfully, numbers: {len(src_dataset)}")

print("Start generation")
os.system("nvidia-smi")
new_delexical = []
epochs = len(src_dataset)//batch
for i in range(epochs):
    new_delexical.extend(model.generate(src_dataset[i: i+batch]))
new_delexical.extend(model.generate(src_dataset[epochs*batch:]))

act_slot_pairs = []
ontology = json.load(open("data/ontology.json", 'r', encoding="utf-8"))
for slot in ontology["slots"]:
    if type(ontology["slots"][slot]) == str:
        path = ontology["slots"][slot]
        path = "data" + path[1:]
        with open(path, "r", encoding="utf-8") as f:
            slotvalues = f.readlines()
            slotvalues = list(map(lambda i: i.replace("\n", ""), slotvalues))
        ontology["slots"][slot] = slotvalues
    for act in ["通知", "否定"]:
        act_slot_pairs.append(f"<{act}-{slot}>")

new_text = []
for text in new_delexical:
    if text == "":
        continue
    new_item = {"utt_id": 1, "manual_transcript":"", "asr_1best":"", "semantic":[]}
    idx = 0
    right_idx = 0
    while idx != -1 and right_idx != -1:
        idx = text.find("<")
        right_idx = text.find(">")
        if idx != right_idx:
            txt = text[idx: right_idx+1]
            act_slot_pairs = sorted(act_slot_pairs, key=lambda x: edit_distance(x, txt))
            pair = act_slot_pairs[0]
            id_ = pair.find("-")
            act = "inform" if pair[1:id_] == "通知" else "deny"
            slot = pair[id_+1:-1]
            value = random.choice(ontology["slots"][slot])
            sem = [act, slot, value]
            text = list(text)
            text[idx: right_idx+1] = value
            text = "".join(text)
            new_item["semantic"].append(sem)
    new_item["manual_transcript"] = text
    new_item["asr_1best"] = text
    new_text.append([new_item])

train_ori = json.load(open("data/train_original.json", 'r', encoding="utf-8"))
train_ori.extend(new_text)
json.dump(train_ori, open("data/train.json", "w+", encoding="utf-8"), indent=6, ensure_ascii=False)
print("New file generated")
