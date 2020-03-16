python simple_train.py \
    --teacher_name shufflenetV2-1.0 \
    --dataset cinic10 \
    --name teacher/cinic10_Sh.NetV2-1.0_e140 \
    -b 96 \
    -j 8  \
    --epochs 140 \
    --lr 0.01 \
    --root /data/cinic10 \
    --tensorboard