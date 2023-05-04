DATA_PATH=/data/lsmdc_data/lsmdc_data
CUDA_VISIBLE_DEVICES=1,0 python -m torch.distributed.launch --nproc_per_node=2 Main.py --do_train --eval_in_train --num_thread_reader=8 --epochs=5 --batch_size=128 --n_display=10 --data_path ${DATA_PATH} --features_path ${DATA_PATH}/Compressclips --output_dir ckpts/ckpt_lsmdc_retrieval_looseType --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 --datatype lsmdc --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32
