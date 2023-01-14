import os
import matplotlib.pyplot as plt
import pickle

exp_dir = 'exp'
files = ['BERT-0.001', 'WORD2V-LSTM']
labels = ['BERT', 'BASE']
results = []
color = ['r', 'g']
for file in files:
    file = os.path.join(exp_dir, file)
    with open(file, 'rb') as f:
        result = pickle.load(f)
    results.append(result)
fig, ax1 = plt.subplots()
ax1.set_xlabel('Step index')
ax1.set_ylabel('Train loss')
ax2 = plt.twinx(ax1)
ax2.set_ylabel('Val acc')
inst = []
for i, result in enumerate(results):
    train_plot = result[0]
    val_plot = result[1]
    train_step_idx, train_loss = list(zip(*train_plot))
    val_step_idx, val_acc = list(zip(*val_plot))
    inst1 = ax1.plot(train_step_idx, train_loss, '-', color=color[i], label=labels[i] + '-loss')
    inst2 = ax2.plot(val_step_idx, val_acc, ':', color=color[i], label=labels[i] + '-acc')
    inst += inst1 + inst2
legends = [line.get_label() for line in inst]
plt.legend(inst, legends, loc='center right', frameon=True)
plt.savefig('tmp.png')