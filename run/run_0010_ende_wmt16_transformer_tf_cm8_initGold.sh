source /home/dawna/tts/qd212/anaconda2/etc/profile.d/conda.sh
conda activate p39_pt17_c10_fs

export CUDA_VISIBLE_DEVICES=0

# ------------------------ MODE --------------------------
MODE=translate # train translate average
tag=e68

EXP_DIR=/home/dawna/tts/qd212/models/fairseq/
cd $EXP_DIR

exp_name=transformer_tf_cm8_initGold_asup
save_dir=checkpoints/en_de_wmt16/$exp_name

init_model=checkpoints/en_de_wmt16/transformer_gold/wmt16.en-de.joined-dict.transformer/model.pt
# init_model=checkpoints/en_de_wmt16/transformer_tf_cm8/checkpoint_best.pt


# ------------------------ RUN --------------------------
echo MODE: $MODE
case $MODE in
"train")
fairseq-train \
    data-bin/wmt16_en_de_bpe32k \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16 --update-freq 8 \
    --save-dir $save_dir \
    --finetune-from-model $init_model
;;

"translate")
fairseq-generate \
    data-bin/wmt16_en_de_bpe32k \
    --path $save_dir/checkpoint_best.pt \
    --beam 4 --lenpen 0.6 --remove-bpe > $save_dir/gen.out
;;

"average")
python scripts/average_checkpoints.py \
    --inputs $save_dir \
    --num-epoch-checkpoints 10 \
    --output $save_dir/checkpoint.avg10_${tag}.pt

fairseq-generate \
    data-bin/wmt16_en_de_bpe32k \
    --path $save_dir/checkpoint.avg10_${tag}.pt \
    --beam 4 --lenpen 0.6 --remove-bpe > $save_dir/gen_avg10_${tag}.out
;;

"bleu")
# bash scripts/compound_split_bleu.sh gen.out
bash scripts/sacrebleu.sh wmt14/full en de $save_dir/gen.out
;;
esac