import io
import gc
import glob
import random
import progressbar
import numpy as np

import chainer
from chainer import cuda


def count_lines(path):
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        return sum([1 for _ in f])
 
def load_data(path, word_ids):
    n_lines = count_lines(path)
    data = []
    i=0
    with open(path) as f:
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        for line in f:
            line=line.split('\t')
            for sent in line:
                words = sent.strip().split()
                if len(words)<3:
                    continue
                words = np.array([word_ids.get(word.lower(), 0) for word in words])
                data.append(words)
    return data

def build_data(path, sub, half, minl, maxl, word_ids, id_vec_, seed_):
    global id_vec
    global seed
    seed=seed_
    id_vec = id_vec_
    data=[]
    if path.split('/')[-1]=='*':
        bar = progressbar.ProgressBar()
        file_list = glob.glob(path)
        for i,p in bar(enumerate(file_list), max_value=len(file_list)):
            d = load_data(p, word_ids)
            data.extend(d)
#            if len(data)>4500:
#                print()
#                data=data[:10000]
#                break
    else:
        print(path)
        data = load_data(path,word_ids)
    if sub:
        Sum=0
        for sent in data:
            if len(sent)==1:
                continue
            sections=[]
            while Sum<len(sent):
                sections.append(random.randint(1,len(sent)//2))
                Sum += sections[-1]
            del sections[-1]
            sections.append(len(sent)-sum(sections))
            sections = np.cumsum(sections)
            seqs = np.split(sent,sections)
            del seqs[-1]
    return data

 
def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    word_dict['<unk>']=0
    i=0
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                i+=1
                word_dict[word] = i 
    return word_dict

def get_glove(glove_path, UNK):
    # create word_vec with glove vectors
    word_ids = {}
    id_vec = {}
    word_ids['<unk>'] = 0
    id_vec[0] = UNK.astype(np.float32)
    with open(glove_path) as f:
        for i,line in enumerate(f):
            word, vec = line.split(' ', 1)
            if not word in word_ids:
                word_ids[word] = i+1
                id_vec[i+1] = np.array(list(map(float, vec.split()))).astype(np.float32)
    return id_vec, word_ids

def build_vocab(glove_path, UNK):
    #word_dict = get_word_dict(sentences)
    id_vec, word_ids = get_glove(glove_path, UNK)
    return id_vec, word_ids

def load_vocab(vocab_path):
    with open(vocab_path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 1 for i, line in enumerate(f)}
    word_ids['UNK'] = 0
    return word_ids

def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            concat = np.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                     for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = np.split(concat_dev, sections)
            return batch_dev
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                     for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev
    
    a = [[id_vec[batch[i][j]] for j in range(len(batch[i]))] for i in range(len(batch))]  
    xs = to_device_batch(a)
    return {'xs':xs}

def convert_CSC(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            concat = np.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                     for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = np.split(concat_dev, sections)
            return batch_dev
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                     for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    global seed
    s1=[]
    s2=[]
    s = [[id_vec.get(batch[i][j], 0) for j in range(len(batch[i]))] for i in range(len(batch))] 
    s = to_device_batch(s)
    for i in range(len(s)):
    #    random.seed(seed)
    #    seed+=1
        cutpoint = random.randint(1,len(s[i])-1)
        s1.append(s[i][:cutpoint])
        s2.append(s[i][cutpoint:])
    return {'xs':s1, 'ys':s2}
