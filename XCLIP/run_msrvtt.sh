# ViT-B/32
job_name="MSRVTT"
python -m torch.distributed.launch --nproc_per_node=2 \
    Main.py --do_train --num_thread_reader=8 \
    --lr 1e-4 --batch_size=256  --batch_size_val 64 \
    --epochs=5  --n_display=50 \
    --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
    --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/MSRVTT_data.json \
    --features_path ${DATA_PATH}/resized_videos \
    --output_dir ckpts/${job_name} \
    --max_words 32 --max_frames 12 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}
