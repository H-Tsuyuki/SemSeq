#!/bin/bash
DATA_FILES_BIG="/home/tsuyuki/data/umbc/tokenized/txt_tokenized/umbc"
VOCAB_PATH="/home/tsuyuki/data/GloVe2/glove.840B.300d.txt"
W2ID_PATH="/home/tsuyuki/data/GloVe2/glv_word_ids.pkl" 
ID2V_PATH="/home/tsuyuki/data/GloVe2/glv_id_vec.pkl"

OUTDIR_="../results/QT_con2_sub1-100gru_NoDirectionQTmask_results"
if [ ! -d $OUTDIR_ ];then
    mkdir $OUTDIR_
fi
OUTDIR_=$OUTDIR_"/result"

for i in {0..9}; do
    echo $(($i+1))
    DATA_FILES="$DATA_FILES_BIG$i/*"
    LOADDIR=$OUTDIR_$i
    OUTDIR="$OUTDIR_$(($i+1))"
    if [ ! -d $OUTDIR ];then
        mkdir $OUTDIR
    fi
    python ../train.py \
        --datapath=$DATA_FILES \
        --outputdir=$OUTDIR \
        --save_model="model$(($i+1))" \
        --save_optimizer="optimizer$(($i+1))" \
        --load_model="$LOADDIR/model$i" \
        --load_optimizer="$LOADDIR/optimizer$i" \
        --vocab_path=$VOCAB_PATH \
        --glv_word_ids=$W2ID_PATH \
        --glv_id_vec=$ID2V_PATH \
        --batch_size=400 \
        --enc_dim=1200 \
        --ENC1='gru' \
        --ENC2='gru' \
        --QTtype='NoDirectionQT' \
        --glove \
        --minl=1  \
        --maxl=100  \
        --gpu=3 \
        --log=1000 \
	--number=$(($i+1)) \
        > "$OUTDIR/out.txt"
        #--sub  \
done

python sample.py
