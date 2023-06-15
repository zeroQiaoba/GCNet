# GCNet: Graph Completion Network for Incomplete Multimodal Learning in Conversation

Correspondence to: 
  - Zheng Lian (lianzheng2016@ia.ac.cn)
  - Licai Sun (sunlicai2019@ia.ac.cn)
  - Lan Chen (chenlanjojo@gmail.com)

## Paper
[**GCNet: Graph Completion Network for Incomplete Multimodal Learning in Conversation**](https://arxiv.org/abs/2203.02177)<br>
Zheng Lian, Lan Chen, Licai Sun, Bin Liu, Jianhua Tao<br>
IEEE Transactions on pattern analysis and machine intelligence, 2022

Please cite our paper if you find our work useful for your research:

```tex
@article{lian2022gcnet,
  title={GCNet: Graph Completion Network for Incomplete Multimodal Learning in Conversation},
  author={Lian, Zheng and Chen, Lan and Sun, Licai and Liu, Bin and Tao, Jianhua},
  journal={IEEE Transactions on pattern analysis and machine intelligence},
  year={2022},
  publisher={IEEE}
}
```

## Usage (Choose IEMOCAP-Six for Example)

### Prerequisites
- Python 3.8
- CUDA 10.2
- pytorch ==1.8.0
- torchvision == 0.9.0
- torch_geometric == 2.0.1
- fairseq == 0.10.1
- transformers==4.5.1
- pandas == 1.2.5

(see requirements.txt for more details)


### Pretrained model

```shell
## for lexical feature extraction
https://huggingface.co/microsoft/deberta-large/tree/main  -> ../tools/transformers/deberta-large

## for acoustic feature extraction
https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt  -> ../tools/wav2vec

## for face extractor (OpenFace-win)
https://drive.google.com/file/d/1-O8epcTDYCrRUU_mtXgjrS3OWA4HTp0-/view?usp=share_link  -> ./OpenFace_2.2.0_win_x64

## for visual feature extraction
https://drive.google.com/file/d/1wT2h5sz22SaEL4YTBwTIB3WoL4HUvg5B/view?usp=share_link ->  ../tools/manet

## using ffmpeg for sub-video extraction
https://ffmpeg.org/download.html#build-linux ->  ../tools/ffmpeg-4.4.1-i686-static
```



### Datasets

~~~~shell
# download IEMOCAP dataset and put it into ../emotion-data/IEMOCAP
https://sail.usc.edu/iemocap/iemocap_release.htm   ->   ../emotion-data/IEMOCAP

# whole video -> subvideo
python preprocess.py split_video_by_start_end_IEMOCAP

# subvideo -> detect face
python detect.py --model='face_detection_yunet_2021sep.onnx' --videofolder='dataset/IEMOCAP/subvideo' --save='dataset/IEMOCAP/subvideofaces' --dataset='IEMOCAP'

# extract visual features
cd feature_extraction/visual
python extract_manet_embedding.py --dataset='IEMOCAPFour' --gpu=0
python preprocess.py feature_compressed_iemocap dataset/IEMOCAP/features/manet dataset/IEMOCAP/features/manet_UTT

# extract acoustic features
python preprocess.py split_audio_from_video_16k 'dataset/IEMOCAP/subvideo' 'dataset/IEMOCAP/subaudio'
cd feature_extraction/audio
python extract_wav2vec_embedding.py --dataset='IEMOCAPFour' --feature_level='UTTERANCE' --gpu=0

# extract textual features
python preprocess.py generate_transcription_files_IEMOCAP
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset='IEMOCAPFour' --feature_level='UTTERANCE' --model_name='deberta-large' --gpu=0

###################################################################
# We also provide pre-extracted multimodal features
IEMOCAP: https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view?usp=share_link  -> ./dataset/IEMOCAP/features
CMUMOSI: https://drive.google.com/file/d/1aJxArYfZsA-uLC0sOwIkjl_0ZWxiyPxj/view?usp=share_link  -> ./dataset/CMUMOSI/features
CMUMOSEI:https://drive.google.com/file/d/1L6oDbtpFW2C4MwL5TQsEflY1WHjtv7L5/view?usp=share_link  -> ./dataset/CMUMOSEI/features
~~~~





### Run GCNet

To evaluate the performance of different methods, we run each experiment 10 times (with different seeds) and report the average values on the test set. 

~~~~shell
cd gcnet
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.2' --windowp=2 --windowf=2 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66
~~~~



### Run MMIN/AE/CRA Baselines

1. change feature format

~~~~shell
cd baseline-mmin
python change_format.py change_feat_format_iemocapsix
~~~~

2. train MMIN model

```shell
# train fully model => for mmin
python train_baseline.py --dataset_mode=iemocapsix_multimodal  --model=utt_fusion --gpu_ids=0 --modality='AVL' --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --cls_layers=128,128 --dropout_rate=0.3 --niter=20 --niter_decay=80 --beta1=0.9 --init_type kaiming --batch_size=256 --lr=1e-3 --run_idx=6 --name=utt_fusion --suffix=iemocapsix_AVL  --output_dim=6

# train mmin model
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapsix_miss  --model=mmin --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128,64 --n_blocks=5 --num_thread=0 --pretrained_path='checkpoints/utt_fusion_iemocapsix_AVL' --ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapsix_MMINTemp
```

3. train AE model

```shell
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapsix_miss  --model=mmin_AE --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapsix_AETemp
```

4. train CRA model

```shell
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapsix_miss  --n_blocks=2 --model=mmin_CRA --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.8 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapsix_CRATemp
```



### Run CPM-Net Baseline

```shell
## differently, CPM-Net runs in another environment, see requirements-cpmnet.txt for more details.
cd baseline-cpmnet

## change feature format
python change_format.py change_feat_format_iemocapsix

## training model
python test_lianzheng.py --dataset='iemocapsix'  --missing-rate=0.2 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=1
```



### Run CCA/DCCA/DCCAE Baselines

```shell
cd baseline-cca

# training with cpmnet-generated data format
please first run ''python change_format.py change_feat_format_iemocapsix'' in baseline-cpmnet

# train CCA
python cca.py   --dataset='iemocapsix' --missing-rate=0.2 --n-components=2

# train DCCA
python dcca.py  --dataset='iemocapsix' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2

# train DCCAE
python dccae.py --dataset='iemocapsix' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2
```



### Other Examples

1. For other datasets, please refer to **run.sh**
2. For parameter turning, please see:

```shell
## dataset: [CMUMOSI, IEMOCAPFour, IEMOCAPSix, CMUMOSEI]
sh run_gcnet.sh [dataset] [gpu_ids]

## dataset: [cmumosi, iemocapfour, iemocapsix, cmumosei]
sh run_mmin.sh [dataset] [gpu_ids]
sh run_ae.sh [dataset] [gpu_ids]
sh run_cra.sh [dataset] [gpu_ids]

## dataset: [cmumosi, iemocapfour, iemocapsix, cmumosei]
sh run_cca.sh [dataset]
sh run_dcca.sh [dataset]
sh run_dccae.sh [dataset]

## run on tf115env37
## dataset: [cmumosi, iemocapfour, iemocapsix, cmumosei]
sh run_cpmnetsub1.sh [dataset]
sh run_cpmnetsub2.sh [dataset]
sh run_cpmnetsub3.sh [dataset]
```



### Acknowledgement

Thanks to [openface](https://github.com/TadasBaltrusaitis/OpenFace), [fairseq](https://github.com/facebookresearch/fairseq), [CPM-Nets](https://github.com/hanmenghan/CPM_Nets), [DialogueGCN](https://github.com/declare-lab/conv-emotion/tree/master/DialogueGCN), [CCA](https://github.com/ashawkey/CCA), [DCCA](https://github.com/Michaelvll/DeepCCA), [MMIN](https://github.com/AIM3-RUC/MMIN/tree/master).
