## A quick demo
Once the installation is done, you can follow the below steps to run a quick demo.
    
    # assume that you are under the root directory of this project,
    # and you have activated your virtual environment if needed.
    wget https://cloudstor.aarnet.edu.au/plus/s/ZSAqNJB96hA71Yf/download -O FCOS_imprv_R_50_FPN_1x.pth
    CUDA_VISIBLE_DEVICES=0 python demo/fcos_demo.py
    
`CUDA_VISIBLE_DEVICES=0 python demo/fcos_demo.py --images-dir ./demo/vid_images/ --res-dir ./demo/results/demo/`


## Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        OUTPUT_DIR inference/uadetrace_val \
        MODEL.WEIGHT FCOS_imprv_R_50_FPN_1x.pth \
        TEST.IMS_PER_BATCH 1    

`CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file configs/fcos/fcos_imprv_R_50_FPN_1x_zzh.yaml OUTPUT_DIR inference_dir/uadetrace_val MODEL.WEIGHT ./training_dir/fcos_imprv_R_50_FPN_1x_zzh/model_0007500.pth TEST.IMS_PER_BATCH 8 `


Please note that:
1) If your model's name is different, please replace `FCOS_imprv_R_50_FPN_1x.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/fcos](configs/fcos)) and `MODEL.WEIGHT` to its weights file.
4) Multi-GPU inference is available, please refer to [#78](https://github.com/tianzhi0549/FCOS/issues/78#issuecomment-526990989).
5) We improved the postprocess efficiency by using multi-label nms (see [#165](https://github.com/tianzhi0549/FCOS/pull/165)), which saves 18ms on average. The inference metric in the following tables has been updated accordingly.

## Training

The following command line will train FCOS_imprv_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x
- multi GPUs
`CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train_net.py --config-file configs/fcos/fcos_imprv_R_50_FPN_1x_zzh.yaml DATALOADER.NUM_WORKERS 2 OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x_zzh ` 

- single GPU
`CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/fcos/fcos_imprv_R_50_FPN_1x_zzh.yaml DATALOADER.NUM_WORKERS 2 OUTPUT_DIR training_dir/test ` 

Note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x.yaml](configs/fcos/fcos_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.
4) If you want to train FCOS on your own dataset, please follow this instruction [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
5) Now, training with 8 GPUs and 4 GPUs can have the same performance. Previous performance gap was because we did not synchronize `num_pos` between GPUs when computing loss. 
