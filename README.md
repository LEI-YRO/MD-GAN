# BMDGAN

# Environment and Supported Toolkits

 python 3.9<br>
 pytorch(http://pytorch.org/)<br>
 tensorflow 2.10.0<br>
 munch 2.5.0<br>
 opencv-python 4.4.0.46<br>
 ffmpeg-python 0.2.0<br>
 
# Demo

 1. Download pre-trained models from [BaiduNetdisk](https://pan.baidu.com/s/1N_k_ei-x9REeVHBb6vnNvw?). password: zfxa.<br>
 2. Create the folder expr, which contains the folders :checkpoints, results, samples.
 3. Copy the pre-training files to the expr/checkpoints/BraTS.
 4. To train ADGAN, run the following command：<br>
```bash
  #BraTS2018
   python main.py --mode train --num_domains 2 --w_hpf 0 \
               --lambda_reg 1 --lambda_rec 0.01 --lambda_per class 0.02 --lambda_l1 100 \
               --train_img_dir data/BraTS/train \
               --val_img_dir data/BraTS/val
```
 5. Test ADGAN by running the following command：<br>
```bash
 #BraTS2018
 python main.py --mode sample --num_domains 2 --resume_iter 0 --w_hpf 0 \
               --checkpoint_dir expr/checkpoints/BraTS \
               --result_dir expr/results/BraTS \
               --src_dir assets/BraTS/src \
               --ref_dir assets/BraTS/ref
```
# Notes
1. The implementation of proposed BMDGAN model is based on StarGAN V2(https://github.com/clovaai/stargan-v2) and ADGAN(https://github.com/LEI-YRO/ADGAN). 
2. To facilitate processing, some image data were uploaded, which were derived from the dataset BraTS2018.
3. If you want to train a custom dataset, the file processing is the same as BraTS.
4. For smooth training of the network, it is recommended that the image naming does not contain any modal nouns.
