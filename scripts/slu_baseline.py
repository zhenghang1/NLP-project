#coding=utf8
import sys, os, time, gc
import json
import matplotlib.pyplot as plt
import pickle

from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example, DevExample, TestExample
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging
from model.bert_tagging import BERTTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print(f">>>> DEVICE: {device}")
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
exp_name = []
# HERE
if not args.segmentation:
    Example.configuration(args.dataroot, train_path=train_path, embedding_path=args.embedding, segmentation=False)
    DevExample.configuration(segmentation=False)
else:
    Example.configuration(args.dataroot, train_path=train_path, embedding_path=args.embedding, segmentation=True)
    DevExample.configuration(segmentation=True)
    print(">>>> Using JIEBA segmentation.")
    exp_name.append('JIEBA')

if not (args.testing or args.inference):
    train_dataset = Example.load_dataset(train_path)
    dev_dataset = DevExample.load_dataset(dev_path)
    print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
elif args.testing:
    dev_dataset = DevExample.load_dataset(dev_path)
    print("Dataset size: dev -> %d" % (len(dev_dataset)))
else:
    test_dataset = TestExample.load_dataset(test_path)
    print("Dataset size: test -> %d" % (len(test_dataset)))
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

if args.embedding == 'bert_model':
    print(">>>> Using BERT model.")
    model = BERTTagging(args).to(device)
    exp_name.append('BERT')
elif args.embedding == 'bert_as_embed':
    print(">>>> Using BERT as embedding.")
    args.bert_as_embed = True
    model = BERTTagging(args).to(device)
    exp_name.append('BERT_EMBED')
elif args.embedding == 'bert_model_with_rnn':
    print(">>>> Using BERT model witn rnn.")
    args.use_rnn = True
    model = BERTTagging(args).to(device)
    exp_name.append('BERT_WITH_RNN')
else:
    print(f">>>> Using {args.encoder_cell} model.")
    model = SLUTagging(args).to(device)
    Example.embedding.load_embeddings(model.word_embed, Example.word_vocab, args.embedding, device=device)
    exp_name.append(args.embedding[:8].upper())
    exp_name.append(args.encoder_cell.upper())


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def test():
    model.eval()
    dataset = test_dataset
    predictions = []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i:i + args.batch_size]
            for ex in cur_dataset:
                print('DEBUG: ', ex.utt)
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.inference(Example.label_vocab, current_batch)
            predictions.extend(pred)
    print(predictions)
    output = []
    for i in range(len(predictions)):
        curr_result = {}
        curr_result["utt_id"] = 1
        # HERE
        if not args.segmentation:
            curr_result["asr_1best"] = dataset[i].utt
        else:
            curr_result["asr_1best"] = dataset[i].utt_uncut
        curr_result["pred"] = []
        for action_slot in predictions[i]:
            curr_result["pred"].append(action_slot.split('-'))
        output.append([curr_result])
    print(output)
    with open('data/test_filled.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    if choice == 'train':
        dataset = train_dataset
    elif choice == 'dev':
        dataset = dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i:i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


if not (args.testing or args.inference):
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' %
          (-1, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
    step_idx = 0
    train_loss_plot = []
    val_acc_plot = []
    val_acc_plot.append((0, dev_acc))
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j:j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
            step_idx += 1
            train_loss_plot.append((step_idx, loss.item()))
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' %
              (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print(
            'Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' %
            (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        val_acc_plot.append((step_idx, dev_acc))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result[
                'iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open(args.model, 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' %
                  (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' %
          (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'],
           best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))

    result_dir = 'exp'
    exp_name.append(str(args.lr))
    exp_name = '-'.join(exp_name)
    result = [train_loss_plot, val_acc_plot]
    os.makedirs(result_dir, exist_ok=True)
    result_name = os.path.join(result_dir, exp_name)
    with open(result_name, 'wb') as f:
        pickle.dump(result, f)

elif args.testing:
    check_point = torch.load(args.model)
    model.load_state_dict(check_point['model'])
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" %
          (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'],
           dev_fscore['fscore']))
elif args.inference:
    check_point = torch.load(args.model)
    model.load_state_dict(check_point['model'])
    start_time = time.time()
    test()
