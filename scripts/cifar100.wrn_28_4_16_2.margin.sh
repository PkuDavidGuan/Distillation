python3 train.py \
    --dataset cifar100 \
    --name cifar100.wrn_28_4_16_2.margin3 \
    --teacher_name wrn-28-4 \
    --student_name wrn-16-2 \
    --epochs 200 \
    --preReLU \
    --kd_method margin_ReLU \
    --teacher_model runs/teacher/cifar100_wrn_28_4_e200/model_best.pth.tar \
    --tensorboard