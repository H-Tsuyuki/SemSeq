import pickle 
import numpy as np
import time
import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.backends import cuda

UNK = 0
EOS = 1

def sequence_embed(embed, xs, dropout=0.):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs

def block_embed(xs):
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]

class GRUEncoder(chainer.Chain):
    def __init__(self, config):
        super(GRUEncoder, self).__init__()
        
        self.n_words = config['n_words']
        self.word_emb_dim = config['word_emb_dim']
        self.dpout_word = config['dpout_word']
        self.enc_dim = config['enc_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.dpout_enc = config['dpout_enc']
        self.glove = config['glove']

        with self.init_scope():
            if not self.glove:
                self.embed = L.EmbedID(self.n_words, self.word_emb_dim)
            self.encoder = L.NStepGRU(self.n_enc_layers, self.word_emb_dim, self.enc_dim, self.dpout_enc)

    def __call__(self, xs):
        xs = tuple([chainer.Variable(xs[i]) for i in range(len(xs))])
        if not self.glove:
            xs = sequence_embed(self.embed, xs, self.dpout_word)        
        embs, _ = self.encoder(None, xs)
        return embs[-1]

class BiGRUEncoder(chainer.Chain):
    def __init__(self, config):
        super(BiGRUEncoder, self).__init__()
        
        self.n_words = config['n_words']
        self.word_emb_dim = config['word_emb_dim']
        self.dpout_word = config['dpout_word']
        self.enc_dim = config['enc_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.dpout_enc = config['dpout_enc']
        self.glove = config['glove']

        self.n_units=self.enc_dim//2

        with self.init_scope():
            if not self.glove:
                self.embed = L.EmbedID(self.n_words, self.word_emb_dim)
            self.encoder = L.NStepBiLSTM(self.n_enc_layers, self.word_emb_dim, self.n_units, self.dpout_enc)

    def __call__(self, xs):
        xs = tuple([chainer.Variable(xs[i]) for i in range(len(xs))])
        if not self.glove:
            xs = sequence_embed(self.embed, xs, self.dpout_word)        
        _, _, ys = self.encoder(None, None, xs)
        embs=F.pad_sequence(ys,padding=0.0)
        embs=F.max(embs,1)
        return embs

class BOWEncoder(chainer.Chain):
    def __init__(self, config):
        super(BOWEncoder, self).__init__()
        
        self.n_words = config['n_words']
        self.word_emb_dim = config['word_emb_dim']
        self.dpout_word = config['dpout_word']
        self.enc_dim = config['enc_dim']
        self.dpout_enc = config['dpout_enc']
        self.glove = config['glove']

        with self.init_scope():
            self.l1 = L.Linear(None, self.enc_dim)

    def __call__(self, xs):
        xs = chainer.dataset.convert.concat_examples(xs, padding=0)
        xs = F.transpose(xs,(1,0,2))
        for i in range(len(xs)):
            if i==0:
                hs = self.l1(xs[0])
                hs = F.expand_dims(hs,0)
            else:
                hw = self.l1(xs[i])
                hw = F.expand_dims(hw,0)
                hs = F.concat((hs,hw), axis=0)
        h=F.mean(hs,axis=0)
        return h

class CNNEncoder(chainer.Chain):
    def __init__(self, config):
        super(CNNEncoder, self).__init__()
        
        self.n_words = config['n_words']
        self.word_emb_dim = config['word_emb_dim']
        self.dpout_word = config['dpout_word']
        self.enc_dim = config['enc_dim']
        self.dpout_enc = config['dpout_enc']
        self.glove = config['glove']
        self.out_units = self.enc_dim//3
        with self.init_scope():
            self.cnn_w3 = L.Convolution2D(
                1, self.out_units, ksize=(3, self.word_emb_dim), stride=1, pad=(2, 0),
                nobias=True)
            self.cnn_w4 = L.Convolution2D(
                1, self.out_units, ksize=(4, self.word_emb_dim), stride=1, pad=(3, 0),
                nobias=True)
            self.cnn_w5 = L.Convolution2D(
                1, self.out_units, ksize=(5, self.word_emb_dim), stride=1, pad=(4, 0),
                nobias=True)

    def __call__(self, xs):
        xs = chainer.dataset.convert.concat_examples(xs, padding=0)
        xs = xs[:,None,:,:]
        h_w3 = F.max(self.cnn_w3(xs), axis=2)
        h_w4 = F.max(self.cnn_w4(xs), axis=2)
        h_w5 = F.max(self.cnn_w5(xs), axis=2)
        h = F.concat([h_w3,h_w4,h_w5],axis=1)
        h = F.dropout(F.relu(h),ratio=self.dpout_enc)
        h = F.squeeze(h)
        return h

class HConvNet(chainer.Chain):
    def __init__(self, config):
        super(HConvNet, self).__init__()
        self.n_words = config['n_words']
        self.word_emb_dim = config['word_emb_dim']
        self.dpout_word = config['dpout_word']
        self.enc_dim = config['enc_dim']
        self.dpout_enc = config['dpout_enc']
        self.glove = config['glove']
        self.out_units = self.enc_dim//4

        with self.init_scope():
            self.conv1 = L.Convolution1D(
                self.word_emb_dim, self.out_units, ksize=3, stride=1, pad=1,
                nobias=True)
            self.conv2 = L.Convolution1D(
                self.out_units, self.out_units, ksize=3, stride=1, pad=1,
                nobias=True)
            self.conv3 = L.Convolution1D(
                self.out_units, self.out_units, ksize=3, stride=1, pad=1,
                nobias=True)
            self.conv4 = L.Convolution1D(
                self.out_units, self.out_units, ksize=3, stride=1, pad=1,
                nobias=True)

    def __call__(self, xs):
        xs = chainer.dataset.convert.concat_examples(xs, padding=0)
        xs = F.transpose(xs,(0,2,1))
        
        sent = F.relu(self.conv1(xs))
        u1 = F.max(sent,axis=2)
        sent = F.relu(self.conv2(sent))
        u2 = F.max(sent,axis=2)
        sent = F.relu(self.conv3(sent))
        u3 = F.max(sent,axis=2)
        sent = F.relu(self.conv4(sent))
        u4 = F.max(sent,axis=2)
        
        h = F.concat([u1,u2,u3,u4], axis=1)
        h = F.dropout(h,ratio=self.dpout_enc)
        return h


class CNN_NgramEncoder(chainer.Chain):
    def __init__(self, config):
        super(CNN_NgramEncoder, self).__init__()
        
        self.n_words = config['n_words']
        self.word_emb_dim = config['word_emb_dim']
        self.dpout_word = config['dpout_word']
        self.enc_dim = config['enc_dim']
        self.dpout_enc = config['dpout_enc']
        self.glove = config['glove']
        self.out_units = self.enc_dim//6

        with self.init_scope():
            self.cnn_w3_s1 = L.Convolution2D(
                1, self.out_units, ksize=(3, self.word_emb_dim), stride=1, pad=(2, 0),
                nobias=True)
            self.cnn_w4_s1 = L.Convolution2D(
                1, self.out_units, ksize=(4, self.word_emb_dim), stride=1, pad=(3, 0),
                nobias=True)
            self.cnn_w5_s1 = L.Convolution2D(
                1, self.out_units, ksize=(5, self.word_emb_dim), stride=1, pad=(4, 0),
                nobias=True)
            
            self.cnn_w3_s2 = L.Convolution2D(
                1, self.out_units, ksize=(3, self.word_emb_dim), stride=1, pad=(2, 0),
                nobias=True)
            self.cnn_w4_s2 = L.Convolution2D(
                1, self.out_units, ksize=(4, self.word_emb_dim), stride=1, pad=(3, 0),
                nobias=True)
            self.cnn_w5_s2 = L.Convolution2D(
                1, self.out_units, ksize=(5, self.word_emb_dim), stride=1, pad=(4, 0),
                nobias=True)

    def __call__(self, xs):
        xs = chainer.dataset.convert.concat_examples(xs, padding=0)
        xs = xs[:,None,:,:]
        
        h_w3_s1 = self.cnn_w3_s1(xs)
        h_w3_s2 = self.cnn_w3_s2(xs)
        h_w3_s1 = F.squeeze(F.transpose(h_w3_s1,(0,2,1,3)))
        h_w3_s2 = F.squeeze(F.transpose(h_w3_s2,(0,2,1,3)))
        from IPython.core.debugger import Pdb; Pdb().set_trace()
        hout_w3 = F.matmul(h_w3_s1, h_w3_s2, transb=True)
        h_w3_s1 = F.max(h_w3_s1, axis=2)
        h_w3_s2 = F.max(h_w3_s2, axis=2)
        
        h_w4_s1 = self.cnn_w4_s1(xs)
        h_w4_s2 = self.cnn_w4_s2(xs)
        h_w4_s1 = F.squeeze(F.transpose(h_w4_s1,(0,2,1,3)))
        h_w4_s2 = F.squeeze(F.transpose(h_w4_s2,(0,2,1,3)))
        hout_w4 = F.matmul(h_w4_s1, h_w4_s2, transb=True)
        h_w4_s1 = F.max(h_w4_s1, axis=2)
        h_w4_s2 = F.max(h_w4_s2, axis=2)

        h_w5_s1 = self.cnn_w5_s1(xs)
        h_w5_s2 = self.cnn_w5_s2(xs)
        h_w5_s1 = F.squeeze(F.transpose(h_w5_s1,(0,2,1,3)))
        h_w5_s2 = F.squeeze(F.transpose(h_w5_s2,(0,2,1,3)))
        hout_w5 = F.matmul(h_w5_s1, h_w5_s2, transb=True)
        h_w5_s1 = F.max(h_w5_s1, axis=2)
        h_w5_s2 = F.max(h_w5_s2, axis=2)


        h = F.concat([h_w3_s1, h_w4_s1, h_w5_s1, h_w3_s2, h_w4_s2, h_w5_s2],axis=1)
        h = F.dropout(F.relu(h),ratio=self.dpout_enc)
        h = F.squeeze(h)
        return h

class Quickthought(chainer.Chain):

    def __init__(self, encoder1, encoder2, config):
        super(Quickthought, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2
        
    def __call__(self, xs):
        bsize = len(xs)
        
        u = self.encoder1(xs)
        v = self.encoder2(xs)
        hout=F.matmul(u, v, transb=True) 

        mask = chainer.Variable(self.xp.identity(bsize,dtype=self.xp.float32))
        hout = hout-(hout*mask)

        targets_np = self.xp.zeros((bsize, bsize),dtype=self.xp.float32)
        ctxt_sent_pos = [-1,1]
        for ctxt_pos in ctxt_sent_pos:
            targets_np += self.xp.eye(bsize, k=ctxt_pos, dtype=self.xp.float32)
        targets_np_sum = self.xp.sum(targets_np, axis=1, dtype=self.xp.float32,keepdims=True)
        targets_np = targets_np/targets_np_sum
        t = chainer.Variable(targets_np)

        hout = F.log_softmax(hout)
        loss = -F.sum(t*hout)/bsize

        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs 

    def set_w2v_dict(self, word_ids_path, id_vec_path):
        with open(word_ids_path, 'rb') as f:
            self.word_ids = pickle.load(f)
        with open(id_vec_path, 'rb') as f:
            self.id_vec = pickle.load(f)
    
    def convert(self, batch, device, glove=True):
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
        if glove:
            
            batch = [[self.word_ids.get(word, 0) for word in sent] for sent in batch]
            batch = [[self.id_vec.get(id, 0) for id in sent] for sent in batch]
        else:
            UNK=0
            batch = [[self.word_ids.get(word, UNK).astype(np.float32) for word in sent] for sent in batch]
        return to_device_batch([x for x in batch])

class QT_all(chainer.Chain):

    def __init__(self, encoder1, encoder2, config):
        super(QT_all, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2
        
    def __call__(self, xs):
        bsize = len(xs)
        
        u = self.encoder1(xs)
        v = self.encoder2(xs)
        hout=F.matmul(u, v, transb=True) 

        #mask = chainer.Variable(self.xp.identity(bsize,dtype=self.xp.float32))
        #hout = hout-(hout*mask)

        targets_np = self.xp.zeros((bsize, bsize),dtype=self.xp.float32)
        ctxt_sent_pos = [1,2] 
        #ctxt_sent_pos = [i for i in range(-(3-1),3)] 
        #ctxt_sent_pos.remove(0)
        for ctxt_pos in ctxt_sent_pos:
            targets_np += self.xp.eye(bsize, k=ctxt_pos, dtype=self.xp.float32)*abs(1/ctxt_pos)
        targets_np_sum = self.xp.sum(targets_np, axis=1, dtype=self.xp.float32,keepdims=True)
        targets_np = targets_np/targets_np_sum
        t = chainer.Variable(targets_np)

        hout = F.log_softmax(hout)
        loss = -F.sum(t*hout)/bsize

        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs 

    def set_w2v_dict(self, word_ids_path, id_vec_path):
        with open(word_ids_path, 'rb') as f:
            self.word_ids = pickle.load(f)
        with open(id_vec_path, 'rb') as f:
            self.id_vec = pickle.load(f)
    
    def convert(self, batch, device, glove=True):
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
        if glove:
            
            batch = [[self.word_ids.get(word, 0) for word in sent] for sent in batch]
            batch = [[self.id_vec.get(id, 0) for id in sent] for sent in batch]
        else:
            UNK=0
            batch = [[self.word_ids.get(word, UNK).astype(np.float32) for word in sent] for sent in batch]
        return to_device_batch([x for x in batch])




class Quickthought_T(chainer.Chain):

    def __init__(self, encoder1, encoder2, config):
        super(Quickthought_T, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2
        
    def __call__(self, xs):
        bsize = len(xs)
        
        u = self.encoder1(xs)
        v = self.encoder2(xs)
        hout=F.matmul(u, v, transb=True) 
        hout_T=F.transpose(hout,(1,0))
        #mask = chainer.Variable(self.xp.identity(bsize,dtype=self.xp.float32))
        #hout = hout-(hout*mask)

        targets_np = self.xp.zeros((bsize, bsize),dtype=self.xp.float32)
        ctxt_sent_pos = [-1,1]
        for ctxt_pos in ctxt_sent_pos:
            targets_np += self.xp.eye(bsize, k=ctxt_pos, dtype=self.xp.float32)

        targets_np_sum = self.xp.sum(targets_np, axis=1, dtype=self.xp.float32,keepdims=True)
        targets_np = targets_np/targets_np_sum
        t = chainer.Variable(targets_np)
        

        hout = F.log_softmax(hout)
        hout_T = F.log_softmax(hout_T)
        loss1 = -F.sum(t*hout)/bsize
        loss1_T = -F.sum(t*hout_T)/bsize
        loss = loss1+loss1_T
        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs 

    def set_w2v_dict(self, word_ids_path, id_vec_path):
        with open(word_ids_path, 'rb') as f:
            self.word_ids = pickle.load(f)
        with open(id_vec_path, 'rb') as f:
            self.id_vec = pickle.load(f)
    
    def convert(self, batch, device, glove=True):
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
        if glove:
            
            batch = [[self.word_ids.get(word, 0) for word in sent] for sent in batch]
            batch = [[self.id_vec.get(id, 0) for id in sent] for sent in batch]
        else:
            UNK=0
            batch = [[self.word_ids.get(word, UNK).astype(np.float32) for word in sent] for sent in batch]
        return to_device_batch([x for x in batch])

class NoDirectionQT(chainer.Chain):

    def __init__(self, encoder1, encoder2, config):
        super(NoDirectionQT, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2
        
    def __call__(self, xs):
        bsize = len(xs)
        
        u = self.encoder1(xs)
        v = self.encoder2(xs)
        hout=F.matmul(u, v, transb=True) 
        mask = chainer.Variable(self.xp.identity(bsize,dtype=self.xp.float32))
        hout = hout-(hout*mask)

        t = chainer.Variable(self.xp.arange(bsize-1,dtype=self.xp.int32)+self.xp.array([1],dtype=self.xp.int32))
        loss1 = F.softmax_cross_entropy(hout[:-1],t)
        t = chainer.Variable(self.xp.arange(bsize-1,dtype=self.xp.int32))
        loss2 = F.softmax_cross_entropy(hout[1:],t)
        loss= loss1+loss2
        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs 

    def set_w2v_dict(self, word_ids_path, id_vec_path):
        with open(word_ids_path, 'rb') as f:
            self.word_ids = pickle.load(f)
        with open(id_vec_path, 'rb') as f:
            self.id_vec = pickle.load(f)
    
    def convert(self, batch, device, glove=True):
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
        if glove:
            
            batch = [[self.word_ids.get(word, 0) for word in sent] for sent in batch]
            batch = [[self.id_vec.get(id, 0) for id in sent] for sent in batch]
        else:
            UNK=0
            batch = [[self.word_ids.get(word, UNK).astype(np.float32) for word in sent] for sent in batch]
        return to_device_batch([x for x in batch])



class NoDirection_T_QT(chainer.Chain):

    def __init__(self, encoder1, encoder2, config):
        super(NoDirection_T_QT, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2
        
    def __call__(self, xs):
        bsize = len(xs)
        
        u = self.encoder1(xs)
        v = self.encoder2(xs)
        hout=F.matmul(u, v, transb=True) 

        t = chainer.Variable(self.xp.arange(bsize-1,dtype=self.xp.int32)+self.xp.array([1],dtype=self.xp.int32))
        loss1 = F.softmax_cross_entropy(hout[:-1],t)
        loss1_t = F.softmax_cross_entropy(F.transpose(hout,(1,0))[:-1],t) 

        t = chainer.Variable(self.xp.arange(bsize-1,dtype=self.xp.int32))
        loss2 = F.softmax_cross_entropy(hout[1:],t)
        loss2_t = F.softmax_cross_entropy(F.transpose(hout,(1,0))[1:],t)
        loss= loss1+loss2+loss1_t+loss2_t
        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs 

    def set_w2v_dict(self, word_ids_path, id_vec_path):
        with open(word_ids_path, 'rb') as f:
            self.word_ids = pickle.load(f)
        with open(id_vec_path, 'rb') as f:
            self.id_vec = pickle.load(f)
    
    def convert(self, batch, device, glove=True):
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
        if glove:
            
            batch = [[self.word_ids.get(word, 0) for word in sent] for sent in batch]
            batch = [[self.id_vec.get(id, 0) for id in sent] for sent in batch]
        else:
            UNK=0
            batch = [[self.word_ids.get(word, UNK).astype(np.float32) for word in sent] for sent in batch]
        return to_device_batch([x for x in batch])


class UniQT(chainer.Chain):

    def __init__(self, encoder1, encoder2, config):
        super(UniQT, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2
        
    def __call__(self, xs):
        bsize = len(xs)
        
        u = self.encoder1(xs)
        v = self.encoder2(xs)
        hout=F.matmul(u, v, transb=True) 
        mask = chainer.Variable(self.xp.identity(bsize,dtype=self.xp.float32))
        hout = hout-(hout*mask)
        t = chainer.Variable(self.xp.arange(bsize-1,dtype=self.xp.int32)+self.xp.array([1],dtype=self.xp.int32))
        loss = F.softmax_cross_entropy(hout[:-1],t)


        #t = chainer.Variable(self.xp.arange(bsize-1,dtype=self.xp.int32))
        #loss = F.softmax_cross_entropy(hout,t)
        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs 

    def set_w2v_dict(self, word_ids_path, id_vec_path):
        with open(word_ids_path, 'rb') as f:
            self.word_ids = pickle.load(f)
        with open(id_vec_path, 'rb') as f:
            self.id_vec = pickle.load(f)
    
class BiQT(chainer.Chain):

    def __init__(self, encoder1, encoder2, config):
        super(BiQT, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2
        
    def __call__(self, xs):
        bsize = len(xs)
        
        u = self.encoder1(xs[:-1])
        v = self.encoder2(xs[1:])
        hout=F.matmul(u, v, transb=True) 

        t = chainer.Variable(self.xp.arange(bsize-1,dtype=self.xp.int32))
        loss1 = F.softmax_cross_entropy(hout,t)
        loss1_t = F.softmax_cross_entropy(F.transpose(hout,(1,0)),t)
        loss= loss1+loss1_t

        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs 

    def set_w2v_dict(self, word_ids_path, id_vec_path):
        with open(word_ids_path, 'rb') as f:
            self.word_ids = pickle.load(f)
        with open(id_vec_path, 'rb') as f:
            self.id_vec = pickle.load(f)
    
class TransEQT(chainer.Chain):

    def __init__(self, encoder1, encoder2, config):
        super(TransEQT, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2
        
    def __call__(self, xs):
        bsize = len(xs)
        
        u = self.encoder1(xs)
        v = self.encoder2(xs)
        h=u[:-1]+v[1:]
        h=h[:-1]
        hout=F.matmul(h, u[2:],transb=True) 

        t = chainer.Variable(self.xp.arange(bsize-2,dtype=self.xp.int32))
        loss1 = F.softmax_cross_entropy(hout,t)
        h=v[:-1]+u[1:]
        h=h[:-1]
        hout=F.matmul(h, v[2:],transb=True)
        loss2 = F.softmax_cross_entropy(hout,t)
        loss=loss1+loss2
        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs 

    def set_w2v_dict(self, word_ids_path, id_vec_path):
        with open(word_ids_path, 'rb') as f:
            self.word_ids = pickle.load(f)
        with open(id_vec_path, 'rb') as f:
            self.id_vec = pickle.load(f)
    
class ConsSentC(chainer.Chain):
    def __init__(self, encoder1, encoder2, config):
        super(ConsSentC, self).__init__()
        with self.init_scope():
            self.encoder1 = encoder1
            self.encoder2 = encoder2

    def __call__(self, xs,ys):
        bsize = len(xs)

        u = self.encoder1(xs)
        v = self.encoder2(ys)
        hout=F.matmul(u, v, transb=True)

        t = chainer.Variable(self.xp.arange(bsize,dtype=self.xp.int32))
        loss = F.softmax_cross_entropy(hout,t)
        chainer.report({'loss': loss.data}, self)
        return loss

    def encode(self, sents):
        embs1 = self.encoder1(sents)
        embs2 = self.encoder2(sents)
        embs = F.concat((embs1,embs2),axis=1)
        return embs

class Multi(chainer.Chain):
    def __init__(self, QT):
        super(Multi, self).__init__()
        with self.init_scope():
            self.QT = QT

    def __call__(self, xs):
        loss1 = self.QT(xs)

        xs = self.xp.concatenate(xs)
        sections=[]
        Sum=0
        while Sum<len(xs):
            sections.append(random.randint(10,70))
            Sum += sections[-1]
        del sections[-1]
        sections.append(len(xs)-sum(sections))
        sections = np.cumsum(sections)
        xs = self.xp.split(xs, sections)
        del xs[-1]
        loss2 = self.QT(xs)

        loss=loss1+loss2
        chainer.report({'loss': loss.data}, self)
        return loss
