import torch
import torch.nn as nn
from torch.optim import Adam
import json, gc, time, sys, os
sys.path.append('.')
from utils.initialization import *
from transformers import EncoderDecoderModel, BertTokenizer

batch_size = 64
max_epoch = 80

translation = json.load(open("data/aug/augtable.json", 'r', encoding="utf-8"))
src_dataset = []
tgt_dataset = []
for src, tgt in translation.items():
    src_dataset.append(src)
    tgt_dataset.append(tgt)


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


def set_optimizer(model):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=1e-5)
    return optimizer


set_random_seed(999)
device = set_torch_device(0)

length = len(src_dataset)
index = list(range(length))
random.shuffle(index)
tmp1 = [src_dataset[k] for k in index]
tmp2 = [tgt_dataset[k] for k in index]


train_dataset = tmp1[:int(0.9*length)]
train_labels = tmp2[:int(0.9*length)]
dev_dataset = tmp1[int(0.9*length):]
dev_labels = tmp2[int(0.9*length):]

model = Augmentor().to(device)

def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    labels = train_labels if choice == 'train' else dev_labels
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            cur_dataset = dataset[i: i + batch_size]
            cur_labels = labels[i: i + batch_size]
            loss = model.forward(cur_dataset, cur_labels)
            total_loss += loss
            count += 1
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / count


num_training_steps = ((len(train_dataset) + batch_size - 1) // batch_size) * max_epoch
print('Total training steps: %d' % (num_training_steps),flush = True)
optimizer = set_optimizer(model)
nsamples, best_result = len(train_dataset), {'loss': 20.}
train_index, step_size = np.arange(nsamples), batch_size
print('Start training ......',flush = True)
for i in range(max_epoch):
    start_time = time.time()
    epoch_loss = 0
    np.random.shuffle(train_index)
    model.train()
    count = 0
    for j in range(0, nsamples, step_size):
        cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
        cur_labels = [train_labels[k] for k in train_index[j: j + step_size]]
        loss = model(cur_dataset, cur_labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        count += 1
    print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count),flush = True)
    torch.cuda.empty_cache()
    gc.collect()

    start_time = time.time()
    dev_loss = decode('dev')
    print('Evaluation: \tEpoch: %d\tTime: %.4f\tEvaluation Loss: %.4f' % (i, time.time() - start_time, dev_loss),flush = True)
    if dev_loss < best_result['loss']:
        best_result['loss'], best_result['iter'] = dev_loss, i
        torch.save({
            'epoch': i, 'model': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, open('model_'+'aug'+'.bin', 'wb'))
        print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f' % (i, dev_loss),flush = True)

print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f' % (best_result['iter'], best_result['dev_loss']),flush = True)
