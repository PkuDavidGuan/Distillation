python3 train.py \
    --dataset cifar10 \
    --name cifar10.wrn_40_2_16_2.sp \
    --teacher_name wrn-40-2 \
    --student_name wrn-16-2 \
    --epochs 200 \
    --kd_method sp \
    --teacher_model runs/teacher/cifar10_wrn_40_2_e200/model_best.pth.tar \
    --tensorboard