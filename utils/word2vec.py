#coding=utf8

import os
import numpy as np
from utils.vocab import PAD, UNK
import torch


class EmbeddingUtils():

    def __init__(self, embedding_file):
        super(EmbeddingUtils, self).__init__()
        if not embedding_file.startswith('bert'):
            self.embedding = {}
            embed_dir = 'embedding'
            embedding_file = os.path.join(embed_dir, embedding_file)
            self.read_from_file(embedding_file)

    def load_embeddings(self, module, vocab, embed, device='cpu'):
        """ Initialize the embedding with glove and char embedding
        """
        emb_size = module.weight.data.size(-1)
        outliers = 0
        missing = 0
        for word in vocab.word2id:
            if word == PAD:  # PAD symbol is always 0-vector
                module.weight.data[vocab[PAD]] = torch.zeros(emb_size, dtype=torch.float, device=device)
                continue
            try:
                word_emb = self.embedding[word]
            except KeyError:
                missing += 1
                if embed == 'embedding/word2vec-768.txt':
                    word_emb = self.embedding[UNK]
                else:
                    word_emb = self.embedding['空']
            module.weight.data[vocab[word]] = torch.tensor(word_emb, dtype=torch.float, device=device)
        print(">>>> Word missing rate in embedding: ", missing / len(vocab.word2id))
        return 1 - outliers / float(len(vocab))

    def read_from_file(self, embedding_file):
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    items = line.split(' ')
                    if len(items) <= 2:
                        continue
                    word = items[0]
                    vector = np.fromstring(' '.join(items[1:]), dtype=float, sep=' ')
                    self.embedding[word] = vector
        except:
            print(
                ">>>> HINT: embedding files should be put into ./embedding directory. DON'T contain dir name in 'args.embedding'."
            )
            exit()
