python3 train.py \
    --dataset cifar100 \
    --name kd_grid_search/0.2_8 \
    --teacher_name wrn-28-4 \
    --student_name wrn-16-2 \
    --epochs 200 \
    --kd_method kd \
    --alpha 0.2 \
    --temperature 8 \
    --teacher_model runs/teacher/cifar100_wrn_28_4_e200/model_best.pth.tar \
    --tensorboard