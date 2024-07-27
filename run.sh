export WANDB_API_KEY=887c77642de1f1be6dbc590391884180cf61c7e6


python main.py \
    --train_path path/to/imagenet/train \
    --val_path path/to/imagenet/val \
    --iterations_meta 500 --bit_w 2 --bit_a 2 \
    --alpha 5 --beta 0.5 --gamma 30000 --thereshold 0.1 \
    --setting=None --use_wandb True






