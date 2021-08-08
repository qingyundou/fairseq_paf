source /home/dawna/tts/qd212/anaconda2/etc/profile.d/conda.sh
conda activate p39_pt17_c10_fs

EXP_DIR=/home/dawna/tts/qd212/models/fairseq/
cd $EXP_DIR

TEXT=wmt16_en_de_bpe32k

fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20