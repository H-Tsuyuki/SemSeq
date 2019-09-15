#!/bin/bash
DATA_FILES_BIG="/home/tsuyuki/SentEval/data/downstream/STS/STS14-en-test"
VOCAB_PATH="/home/tsuyuki/data/GloVe/glove.840B.300d.txt"
W2ID_PATH="/home/tsuyuki/data/GloVe/glv_word_ids.pkl" 
ID2V_PATH="/home/tsuyuki/data/GloVe/glv_id_vec.pkl"
DATA_FILE="STS.input.deft-forum.txt"
DATA_FILE_PATH=$DATA_FILES_BIG"/"$DATA_FILE

OUTDIR_="../finetune_results"
if [ ! -d $OUTDIR_ ];then
    mkdir $OUTDIR_
fi
OUTDIR_=$OUTDIR_"/"$DATA_FILE
if [ ! -d $OUTDIR_ ];then
    mkdir $OUTDIR_
fi

OUTDIR=$OUTDIR_"/QT_con_sub10-70gru_NoDirectionQTmask_results3"
if [ ! -d $OUTDIR ];then
    mkdir $OUTDIR
fi
python ../finetune.py \
    --datapath=$DATA_FILE_PATH \
    --outputdir=$OUTDIR \
    --load_model="/home/tsuyuki/QTsub/results/QT_con_sub10-70gru_NoDirectionQTmask_results3/result10/model10" \
    --save_model="model" \
    --vocab_path=$VOCAB_PATH \
    --glv_word_ids=$W2ID_PATH \
    --glv_id_vec=$ID2V_PATH \
    --batch_size=64 \
    --enc_dim=1200 \
    --n_epochs=5 \
    --ENC1='gru' \
    --ENC2='gru' \
    --QTtype='CSC' \
    --minl=10 \
    --maxl=70 \
    --glove \
    --gpu=2 \
    --log=1 \
    --number=$(($i+1)) \
#    > "$OUTDIR/out.txt"
 #   --sub  \

python sample.py
