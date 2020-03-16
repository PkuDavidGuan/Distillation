python3 train.py \
    --dataset cifar100 \
    --name cifar100.wrn_28_4_16_4.sp \
    --teacher_name wrn-28-4 \
    --student_name wrn-16-4 \
    --epochs 200 \
    --kd_method sp \
    --teacher_model runs/teacher/cifar100_wrn_28_4_e200/model_best.pth.tar \
    --tensorboard