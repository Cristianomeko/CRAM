#!/usr/bin/env bash
#!/usr/bin/env python3
LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

#./train.py --arch="swin_tiny_patch4_window7_224" \
#./train.py --arch="shufflenet_v2_x0_5" \
#../train.py --arch="vanillanet_13" \
../train.py --arch="repvgg" \
    --start-epoch=42 \
    --loss=wpdc \
    --snapshot="snapshot/phase3_wpdc" \
    --param-fp-train='../train.configs/param_all_norm.pkl' \
    --param-fp-val='../train.configs/param_all_norm_val.pkl' \
    --warmup=5 \
    --resume='/home/maxu/yzw/HFCAN/training/snapshot/phase3_wpdc_checkpoint_epoch_41.pth.tar' \
    --opt-style=resample \
    --resample-num=132 \
    --batch-size=128 \
    --base-lr=0.02 \
    --epochs=130 \
    --milestones=21,40,117 \
    --print-freq=50 \
    --devices-id=0,1,2,3 \
    --workers=8 \
    --filelists-train="../train.configs/train_aug_120x120.list.train" \
    --filelists-val="../train.configs/train_aug_120x120.list.val" \
    --root="/home/maxu/Data/train_aug_120x120" \
    --log-file="${LOG_FILE}"
