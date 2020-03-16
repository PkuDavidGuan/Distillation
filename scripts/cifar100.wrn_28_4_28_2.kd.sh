python3 train.py \
    --dataset cifar100 \
    --name cifar100.wrn_28_4_28_2.kd \
    --teacher_name wrn-28-4 \
    --student_name wrn-28-2 \
    --epochs 200 \
    --kd_method kd \
    --temperature 4 \
    --teacher_model runs/teacher/cifar100_wrn_28_4_e200/model_best.pth.tar \
    --tensorboard
