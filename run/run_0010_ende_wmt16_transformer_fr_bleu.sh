# ------------------------ ENV --------------------------
source /home/dawna/tts/qd212/anaconda2/etc/profile.d/conda.sh
conda activate p39_pt17_c10_fs

AIR_FORCE_GPU=0
export MANU_CUDA_DEVICE=0 #note on nausicaa no.2 is no.0
# select gpu when not on air
if [[ "$HOSTNAME" != *"air"* ]]  || [ $AIR_FORCE_GPU -eq 1 ]; then
  X_SGE_CUDA_DEVICE=$MANU_CUDA_DEVICE
fi
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo "on $HOSTNAME, using gpu (no nb means cpu) $CUDA_VISIBLE_DEVICES"

# multiGPU fix
if [ ${#CUDA_VISIBLE_DEVICES} -gt 1 ]; then
  # export MKL_SERVICE_FORCE_INTEL=1
  export MKL_THREADING_LAYER=GNU
fi


# ------------------------ EXP --------------------------
exp_name=0010_transformer_fr_cm8_bleu
# exp_name=asup


# ------------------------ MODE --------------------------
MODE=translate # train translate


# ------------------------ DIR --------------------------
EXP_DIR=/home/dawna/tts/qd212/models/fairseq/
cd $EXP_DIR

save_dir=checkpoints/en_de_wmt16/$exp_name

init_model=checkpoints/en_de_wmt16/transformer_gold/wmt16.en-de.joined-dict.transformer/model.pt


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
    --criterion label_smoothed_cross_entropy_fr --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16 --update-freq 8 \
    --save-dir $save_dir \
    --finetune-from-model $init_model \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 10 --no-epoch-checkpoints \
    > $save_dir/log.txt 2>&1
;;

"average")
python scripts/average_checkpoints.py \
    --inputs $save_dir \
    --num-epoch-checkpoints 10 \
    --output $save_dir/checkpoint.avg10.pt
;;

"translate")
fairseq-generate \
    data-bin/wmt16_en_de_bpe32k \
    --path $save_dir/checkpoint_best.pt \
    --beam 4 --lenpen 0.6 --remove-bpe > $save_dir/gen.out
;;

"bleu")
# bash scripts/compound_split_bleu.sh gen.out
bash scripts/sacrebleu.sh wmt14/full en de $save_dir/gen.out
;;
esac