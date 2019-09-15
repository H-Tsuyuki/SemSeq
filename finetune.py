import os
import sys
import json
import time
import pickle
import argparse
import datetime
import numpy as np

import traceback
import error_notification2

import chainer
from chainer.training import extensions

from utils_finetune import build_data, build_vocab, convert, convert_CSC, load_vocab
from nets_con import Quickthought,QT_all, Quickthought_T, NoDirectionQT, NoDirection_T_QT, UniQT, BiQT, TransEQT, ConsSentC, GRUEncoder, BiGRUEncoder,BOWEncoder, CNNEncoder, HConvNet

def main():

    parser = argparse.ArgumentParser(description='NLI training')
    # paths
    parser.add_argument("--datapath", type=str, default='/home/tsuyuki/data/umbc/tokenized/txt_tokenized/delorme.com_shu.pages_1.txt', help="quora data path")
    parser.add_argument("--outputdir", '-out', type=str, default='test_result', help="Output directory")
    parser.add_argument("--glv_word_ids", type=str, default='/home/tsuyuki/data/GloVe/glv_word_ids.pkl')
    parser.add_argument("--glv_id_vec", type=str, default='/home/tsuyuki/data/GloVe/glv_id_vec.pkl')
    parser.add_argument("--save_model", type=str, default='/home/tsuyuki/QTsub/results/')
    parser.add_argument("--save_optimizer", type=str, default='/home/tsuyuki/QTsub/results/')
    parser.add_argument("--load_model", type=str, default='/home/tsuyuki/QTsub/results/')
    parser.add_argument("--load_optimizer", type=str, default='/home/tsuyuki/QTsub/results/')

    # training
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--dpout_word", type=float, default=0.3, help="word encoder dropout")
    parser.add_argument("--dpout_enc", type=float, default=0.3, help="encoder dropout")
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
    parser.add_argument("--lr", type=float, default=0.0005, help="lr decay")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="lr decay")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="lr decay")
    
    # model
    parser.add_argument("--ENC1", default='gru', choices=['gru','bigru','bow','cnn','hconv'])
    parser.add_argument("--ENC2", default='gru', choices=['gru','bigru','bow','cnn','hconv'])
    parser.add_argument("--QTtype", default='QT',
        choices=['QT','QT_all','QT.T','NoDirectionQT','NoDirection_T_QT','UniQT','BiQT','TransEQT', 'CSC'])
    parser.add_argument("--glove", action='store_true', default=False, help="flag of using glove")
    parser.add_argument("--sub", action='store_true', default=False, help="flag of using glove")
    parser.add_argument("--half", action='store_true', default=False, help="flag of using glove")
    parser.add_argument("--minl", type=int, default=1, help="flag of using glove")
    parser.add_argument("--maxl", type=int, default=50, help="flag of using glove")
    parser.add_argument("--vocab_path", type=str, default="/home/tsuyuki/data/GloVe/glove.840B.300d.txt")
    parser.add_argument("--word_emb_dim", type=int, default=620, help="word embedding dimension")
    parser.add_argument("--enc_dim", type=int, default=1200, help="encoder nhid dimension")
    parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    
    # gpu
    parser.add_argument("--gpu", type=int, default=2, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
   
    #log
    parser.add_argument("--number", type=int, default=0)
    parser.add_argument("--log", type=int, default=100, help='log interval')
    parser.add_argument("--val_interval", '-vallog', type=int, default=10, help='log interval')
    
    params, _ = parser.parse_known_args()
    if params.glove: params.word_emb_dim =300

    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(json.dumps(params.__dict__, indent=2))

    """
    SEED
    """
#    np.random.seed(params.seed)
#    chainer.cuda.cupy.random.seed(params.seed)
#    chainer.global_config.autotune=True
    """
    DATA
    """

    if params.glove:
        if params.glv_id_vec and params.glv_word_ids:
            with open(params.glv_id_vec,'rb') as f:
                id_vec = pickle.load(f)
            with open(params.glv_word_ids,'rb') as f:
                word_ids = pickle.load(f)
        else:
            UNK = np.random.rand(1, params.word_emb_dim)[0]
            id_vec, word_ids = build_vocab(params.vocab_path, UNK)
            with open('dataset/glv_id_vec.pkl','wb') as f:
                pickle.dump(id_vec, f)
            with open('dataset/glv_word_ids.pkl','wb') as f:
                pickle.dump(word_ids, f)
        train = build_data(params.datapath, params.sub, params.half, params.minl, params.maxl, word_ids, id_vec, params.seed)
    else:
        UNK=0
        word_ids = load_vocab(params.vocab_path)
        train, valid, test = build_data(params.datapath, params.sub, word_ids)
    
    print('Train pair size: %d' % len(train))

    """
    MODEL
    """
    # model config
    config_model = {
        'n_words'        :  len(word_ids)         ,
        'word_emb_dim'   :  params.word_emb_dim   ,
        'dpout_word'     :  params.dpout_word     ,
        'enc_dim'        :  params.enc_dim        ,
        'n_enc_layers'   :  params.n_enc_layers   ,
        'dpout_enc'      :  params.dpout_enc      ,
        'bsize'          :  params.batch_size     ,
        'glove'          :  params.glove          ,
    }

    # model
    if params.ENC1=='gru':
        encoder1 = GRUEncoder(config_model)
    elif params.ENC1=='bigru':
        encoder1 = BiGRUEncoder(config_model)
    elif params.ENC1=='bow':
        encoder1 = BOWEncoder(config_model)
    elif params.ENC1=='cnn':
        encoder1 = CNNEncoder(config_model)
    elif params.ENC1=='hconv':
        encoder1 = HConvEncoder(config_model)
    
    if params.ENC2=='gru':
        encoder2 = GRUEncoder(config_model)
    elif params.ENC2=='bigru':
        encoder2 = BiGRUEncoder(config_model)
    elif params.ENC2=='bow':
        encoder2 = BOWEncoder(config_model)
    elif params.ENC2=='cnn':
        encoder2 = CNNEncoder(config_model)
    elif params.ENC2=='hconv':
        encoder2 = HConvEncoder(config_model)
    
    if params.QTtype=='QT':
        model = Quickthought(encoder1, encoder2, config_model)
    elif params.QTtype=='QT_all':
        model = QT_all(encoder1, encoder2, config_model)
    elif params.QTtype=='QT.T':
        model = Quickthought_T(encoder1, encoder2, config_model)
    elif params.QTtype=='NoDirectionQT':
        model = NoDirectionQT(encoder1, encoder2, config_model)
    elif params.QTtype=='NoDirection_T_QT':
        model = NoDirection_T_QT(encoder1, encoder2, config_model)
    elif params.QTtype=='UniQT':
        model = UniQT(encoder1, encoder2, config_model)
    elif params.QTtype=='BiQT':
        model = BiQT(encoder1, encoder2, config_model)
    elif params.QTtype=='TransEQT':
        model = TransEQT(encoder1, encoder2, config_model)
    elif params.QTtype=='CSC':
        model = ConsSentC(encoder1, encoder2, config_model)
    
    if os.path.isfile(params.load_model):
        chainer.serializers.load_npz(params.load_model, model)    
    
    if params.gpu >= 0:
        chainer.backends.cuda.get_device(params.gpu).use()
        model.to_gpu(params.gpu)
    
    # optimizer
    optimizer = chainer.optimizers.Adam(params.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(params.weight_decay)) 
    #optimizer.add_hook(chainer.optimizer.GradientClipping(params.grad_clip))

    if os.path.isfile(params.load_optimizer):
        chainer.serializers.load_npz(params.load_optimizer, optimizer)    
    
    #iterator
    train_iter = chainer.iterators.SerialIterator(train, params.batch_size, True, True)
    
    #updater
    if params.QTtype=='CSC':
        updater = chainer.training.updaters.StandardUpdater(
            train_iter, optimizer, converter=convert_CSC, device=params.gpu)
    else:
        updater = chainer.training.updaters.StandardUpdater(
            train_iter, optimizer, converter=convert, device=params.gpu)

    trainer = chainer.training.Trainer(updater, (params.n_epochs, 'epoch'), out=params.outputdir)
    
#    if params.number==1:
#        chainer.serializers.load_npz('/home/tsuyuki/QTsub/results/QT_con_subgru_results/snapshot-80000', trainer)    
    
    #log
    trainer.extend(extensions.LogReport(trigger=(params.log, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/accuracy', 'elapsed_time']), trigger=(params.log, 'iteration'))
   
    #save
#    trainer.extend(extensions.snapshot(filename='snapshot-{.updater.iteration}'),trigger=(20000, 'iteration'))
#    trainer.extend(extensions.snapshot_object(model, filename=params.save_model+"-{.updater.iteration}"),trigger=(1000, 'iteration'))
    trainer.extend(extensions.snapshot_object(model, filename=params.save_model))
#    trainer.extend(extensions.snapshot_object(optimizer, filename=params.save_optimizer))
    
    trainer.extend(extensions.ProgressBar())

    print('[{}] start training'.format(datetime.datetime.now())) 
    trainer.run()
    with open(params.outputdir + '/' + 'paramslog.txt','w') as f:
        f.write(json.dumps(params.__dict__, indent=2))

    error_notification2.send_on_going_status(str(params.number)+'/407 '+ params.outputdir.split('/')[2])

if __name__ == '__main__':
    try:
        main()
    except: 
        error_notification2.send_error_log(traceback.format_exc())

