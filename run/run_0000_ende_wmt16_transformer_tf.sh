source /home/dawna/tts/qd212/anaconda2/etc/profile.d/conda.sh
conda activate p39_pt17_c10_fs

# -----------------------------

EXP_DIR=/home/dawna/tts/qd212/models/fairseq/
cd $EXP_DIR


# fairseq-train \
#     data-bin/wmt16_en_de_bpe32k \
#     --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
#     --dropout 0.3 --weight-decay 0.0 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 3584 \
#     --fp16

# CUDA_VISIBLE_DEVICES=1 python scripts/average_checkpoints.py \
#     --inputs checkpoints \
#     --num-epoch-checkpoints 5 \
#     --output checkpoints/checkpoint.avg5.pt


# CUDA_VISIBLE_DEVICES=1 fairseq-generate \
#     data-bin/wmt16_en_de_bpe32k \
#     --path checkpoints/checkpoint_best.pt \
#     --beam 4 --lenpen 0.6 --remove-bpe > gen.out

# bash scripts/compound_split_bleu.sh gen.out
# bash scripts/sacrebleu.sh wmt14/full en de gen.out


ckpt_dir=checkpoints/en_de_wmt16/transformer_tf
# fairseq-generate \
#     data-bin/wmt16_en_de_bpe32k \
#     --path $ckpt_dir/checkpoint.avg5.pt \
#     --beam 4 --lenpen 0.6 --remove-bpe > $ckpt_dir/gen_avg5.out

# bash scripts/sacrebleu.sh wmt14/full en de $ckpt_dir/gen_avg5.out






ckpt_dir=checkpoints/en_de_wmt16/transformer_gold
# fairseq-generate \
#     data-bin/wmt16.en-de.joined-dict.newstest2014 \
#     --path $ckpt_dir/wmt16.en-de.joined-dict.transformer/model.pt \
#     --beam 4 --lenpen 0.6 --remove-bpe > $ckpt_dir/gen.out

bash scripts/sacrebleu.sh wmt14/full en de $ckpt_dir/gen.out