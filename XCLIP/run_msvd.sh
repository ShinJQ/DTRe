DATA_PATH=~/data/msvd_data
job_name="MSVD"
python -m torch.distributed.launch --nproc_per_node=2 \
    Main.py --do_train --num_thread_reader=8 \
    --epochs=5 --batch_size=256 --n_display=10 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/Compressclips \
    --output_dir ckpts/${job_name} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 96 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}
