import json, os, math,sys
from nltk.metrics import edit_distance

sys.path.append("")
class Delexical():
    @classmethod
    def load_dataset(cls, data_path="./data/train.json"):
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


class DataAugmentation:
    def __init__(self):
        self.delexical_ex = Delexical.load_dataset()
        self.cluster = dict()
        self.source_str = []
        self.target_str = []   
        self.form_cluster()
        self.rank()

    def diverse_score(self, src, tgt):
        length_penalty = math.exp(-abs((len(src) - len(tgt))/len(src)))
        e_d = edit_distance(tgt, src)
        return e_d * length_penalty

    def form_cluster(self):
        for ex in self.delexical_ex: 
            self.cluster.setdefault(ex.slot, [])
            self.cluster[ex.slot].append(ex.utt)


    def rank(self):
        for ex in self.delexical_ex:
            cluster = set(self.cluster[ex.slot])
            ranking = sorted(cluster, key=lambda tgt: self.diverse_score(ex.utt, tgt), reverse=True)
            if ex.slot == tuple():
                ranking = ranking[:30]
            else:
                ranking = ranking[:len(cluster)//2]
            for i, tgt in enumerate(ranking):
                self.source_str.append(ex.utt+f"#{i+1}")
                self.target_str.append(tgt)
        self.translation = {src: tgt for src, tgt in zip(self.source_str, self.target_str)}


aug = DataAugmentation()
json.dump(aug.translation, open("./data/aug/augtable.json", "w+", encoding="utf-8"), indent=6, ensure_ascii=False)