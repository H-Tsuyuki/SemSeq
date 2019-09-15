#!/bin/bash
DATA_FILES="/home/tsuyuki/data/umbc/tokenized/txt_tokenized/eval/*"
VOCAB_PATH="/home/tsuyuki/data/GloVe/glove.840B.300d.txt"
W2ID_PATH="/home/tsuyuki/data/GloVe/glv_word_ids.pkl" 
ID2V_PATH="/home/tsuyuki/data/GloVe/glv_id_vec.pkl"
LOADMODEL="../results/QT_con_gru_NoDirectionQT_results/result10/model10"
OUTDIR="../test_results/QT_con_gru_NoDirectionQT-subgru_NoDirectionQT"
#OUTDIR="../test_results/a"

if [ ! -d $OUTDIR ];then
    mkdir $OUTDIR
fi
python ../test.py \
    --datapath=$DATA_FILES \
    --outputdir=$OUTDIR \
    --load_model="$LOADMODEL" \
    --vocab_path=$VOCAB_PATH \
    --glv_word_ids=$W2ID_PATH \
    --glv_id_vec=$ID2V_PATH \
    --batch_size=400 \
    --ENC1='gru' \
    --ENC2='gru' \
    --QTtype='NoDirectionQT' \
    --glove \
    --gpu=1 \
    --log=10 \
    > "$OUTDIR/random-usefor_gru.txt"
#    --sub  \

python sample.py
