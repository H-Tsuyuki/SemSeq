#!/bin/bash
DATA_FILES_BIG="/home/tsuyuki/data/umbc/tokenized/txt_tokenized/umbc"
VOCAB_PATH="/home/tsuyuki/data/GloVe/glove.840B.300d.txt"
W2ID_PATH="/home/tsuyuki/data/GloVe/glv_word_ids.pkl" 
ID2V_PATH="/home/tsuyuki/data/GloVe/glv_id_vec.pkl"

OUTDIR_="../results/QT_con_sub1-50gru_NoDrectionQTmask_results"
if [ ! -d $OUTDIR_ ];then
    mkdir $OUTDIR_
fi
OUTDIR_=$OUTDIR_"/result"
i=0
for j in {0..9}; do
        DATA_FILES="$DATA_FILES_BIG$j/*"
    for filename in $DATA_FILES; do
        echo $(($i+1))
        DATA_FILE=$filename
        LOADDIR=$OUTDIR_$i
        OUTDIR="$OUTDIR_$(($i+1))"
        if [ ! -d $OUTDIR ];then
            mkdir $OUTDIR
        fi
        python ../train.py \
            --datapath=$DATA_FILE \
            --outputdir=$OUTDIR \
            --save_model="model$(($i+1))" \
            --save_optimizer="optimizer$(($i+1))" \
            --load_model="$LOADDIR/model$i" \
            --load_optimizer="$LOADDIR/optimizer$i" \
            --vocab_path=$VOCAB_PATH \
            --glv_word_ids=$W2ID_PATH \
            --glv_id_vec=$ID2V_PATH \
            --batch_size=400 \
            --ENC1='gru' \
            --ENC2='gru' \
            --QTtype='NoDirectionQT' \
            --glove \
            --sub  \
            --minl=1 \
            --maxl=50 \
            --gpu=3 \
            --log=100 \
    	    --number=$(($i+1)) \
            > "$OUTDIR/out.txt"
        i=$(($i+1))
    done
done

python sample.py
