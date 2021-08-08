source /home/dawna/tts/qd212/anaconda2/etc/profile.d/conda.sh
conda activate p39_pt17_c10_fs

# -----------------------------

EXP_DIR=/home/dawna/tts/qd212/models/fairseq/
cd $EXP_DIR

exp_name=transformer_tf_cm8
save_dir=checkpoints/en_de_wmt16/$exp_name

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/wmt16_en_de_bpe32k \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16 --update-freq 8 \
    --save-dir $save_dir


# CUDA_VISIBLE_DEVICES=1 python scripts/average_checkpoints.py \
#     --inputs checkpoints \
#     --num-epoch-checkpoints 5 \
#     --output checkpoints/checkpoint.avg5.pt


# fairseq-generate \
#     data-bin/wmt16_en_de_bpe32k \
#     --path $save_dir/checkpoint_best.pt \
#     --beam 4 --lenpen 0.6 --remove-bpe > $save_dir/gen.out

# bash scripts/compound_split_bleu.sh gen.out
# bash scripts/sacrebleu.sh wmt14/full en de gen.out