#####################################################
############ Data preprocess for IEMOCAP ############
#####################################################
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


#####################################################
############ Data preprocess for CMUMOSI ############
#####################################################
# download CMUMOSI dataset and put it into ../emotion-data/CMUMOSI
http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset   ->   ../emotion-data/CMUMOSI
mv ../emotion-data/CMUMOSI/Raw/Video/Segmented dataset/CMUMOSI/subvideo

# subvideo -> detect face
cd feature_extraction/visual
python extract_openface.py --dataset=CMUMOSI --type=videoOne
mv ../emotion-data/CMUMOSI/Raw/features/openface_face dataset/CMUMOSI/openface_face

# extract visual features
cd feature_extraction/visual
python extract_manet_embedding.py --dataset='CMUMOSI' --gpu=0
python preprocess.py feature_compressed dataset/CMUMOSI/features/manet dataset/CMUMOSI/features/manet_UTT

# extract acoustic features
python preprocess.py split_audio_from_video_16k 'dataset/CMUMOSI/subvideo' 'dataset/CMUMOSI/subaudio'
cd feature_extraction/audio
python extract_wav2vec_embedding.py --dataset='CMUMOSI' --feature_level='UTTERANCE' --gpu=0

# extract textual features
python preprocess.py generate_transcription_files_CMUMOSI
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset='CMUMOSI' --feature_level='UTTERANCE' --model_name='deberta-large' --gpu=0


#####################################################
############ Data preprocess for CMUMOSEI ###########
#####################################################
# download CMUMOSI dataset and put it into ../emotion-data/CMUMOSI
http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/   ->   ../emotion-data/CMUMOSEI

# whole video -> subvideo
python preprocess.py split_video_by_start_end_CMUMOSEI
python preprocess.py select_videos_for_cmumosei

# subvideo -> detect face
cd feature_extraction/visual
python extract_openface.py --dataset=CMUMOSEI --type=videoOne
mv ../emotion-data/CMUMOSEI/Raw/features/openface_face dataset/CMUMOSEI/openface_face

# extract visual features
cd feature_extraction/visual
python extract_manet_embedding.py --dataset='CMUMOSEI' --gpu=0
python preprocess.py feature_compressed dataset/CMUMOSEI/features/manet dataset/CMUMOSEI/features/manet_UTT

# extract acoustic features
python preprocess.py split_audio_from_video_16k 'dataset/CMUMOSEI/subvideo' 'dataset/CMUMOSEI/subaudio'
cd feature_extraction/audio
python extract_wav2vec_embedding.py --dataset='CMUMOSEI' --feature_level='UTTERANCE' --gpu=0

# extract textual features
python preprocess.py generate_transcription_files_CMUMOSEI
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset='CMUMOSEI' --feature_level='UTTERANCE' --model_name='deberta-large' --gpu=0


##########################################################################################
############ Train GCNet (test 10 different seeds and report average results) ############
##########################################################################################
cd gcnet
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.0' --windowp=2 --windowf=2 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.1' --windowp=2 --windowf=2 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.2' --windowp=2 --windowf=2 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.3' --windowp=1 --windowf=1 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.4' --windowp=2 --windowf=2 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=100 --mask-type='constant-0.5' --windowp=2 --windowf=2 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.6' --windowp=1 --windowf=1 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --epoch=100 --lr=0.001 --hidden=100 --mask-type='constant-0.7' --windowp=3 --windowf=3 --base-model='LSTM' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=66



###########################################
############ Train MMIN/AE/CRA ############
###########################################
cd baseline-mmin

## Step1: change feature format
python change_format.py change_feat_format_cmumosi
python change_format.py change_feat_format_iemocapfour
python change_format.py change_feat_format_iemocapsix
python change_format.py change_feat_format_cmumosei

## Step2-1: train fully model => for mmin (change dataset_mode, suffix, output_dim)
## Step2-2: train mmin model (change dataset_mode, pretrained_path, suffix)
python train_baseline.py --dataset_mode=cmumosi_multimodal     --model=utt_fusion --gpu_ids=0 --modality='AVL' --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --cls_layers=128,128 --dropout_rate=0.3 --niter=20 --niter_decay=80 --beta1=0.9 --init_type kaiming --batch_size=256 --lr=1e-3 --run_idx=6 --name=utt_fusion --suffix=cmumosi_AVL     --output_dim=1
python train_baseline.py --dataset_mode=iemocapfour_multimodal --model=utt_fusion --gpu_ids=0 --modality='AVL' --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --cls_layers=128,128 --dropout_rate=0.3 --niter=20 --niter_decay=80 --beta1=0.9 --init_type kaiming --batch_size=256 --lr=1e-3 --run_idx=6 --name=utt_fusion --suffix=iemocapfour_AVL --output_dim=4
python train_baseline.py --dataset_mode=iemocapsix_multimodal  --model=utt_fusion --gpu_ids=0 --modality='AVL' --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --cls_layers=128,128 --dropout_rate=0.3 --niter=20 --niter_decay=80 --beta1=0.9 --init_type kaiming --batch_size=256 --lr=1e-3 --run_idx=6 --name=utt_fusion --suffix=iemocapsix_AVL  --output_dim=6
python train_baseline.py --dataset_mode=cmumosei_multimodal    --model=utt_fusion --gpu_ids=0 --modality='AVL' --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --cls_layers=128,128 --dropout_rate=0.3 --niter=20 --niter_decay=80 --beta1=0.9 --init_type kaiming --batch_size=256 --lr=1e-3 --run_idx=6 --name=utt_fusion --suffix=cmumosei_AVL    --output_dim=1

python -u train_miss.py --mask_rate=0.2 --dataset_mode=cmumosi_miss     --model=mmin --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128,64 --n_blocks=5 --num_thread=0 --pretrained_path='checkpoints/utt_fusion_cmumosi_AVL' --ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=cmumosi_MMINTemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapfour_miss --model=mmin --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128,64 --n_blocks=5 --num_thread=0 --pretrained_path='checkpoints/utt_fusion_iemocapfour_AVL' --ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapfour_MMINTemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapsix_miss  --model=mmin --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128,64 --n_blocks=5 --num_thread=0 --pretrained_path='checkpoints/utt_fusion_iemocapsix_AVL' --ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapsix_MMINTemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=cmumosei_miss    --model=mmin --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128,64 --n_blocks=5 --num_thread=0 --pretrained_path='checkpoints/utt_fusion_cmumosei_AVL' --ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=cmumosei_MMINTemp

## Step3: train AE model (change dataset_mode and suffix)
python -u train_miss.py --mask_rate=0.2 --dataset_mode=cmumosi_miss     --model=mmin_AE --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=cmumosi_AETemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapfour_miss --model=mmin_AE --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapfour_AETemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapsix_miss  --model=mmin_AE --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapsix_AETemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=cmumosei_miss    --model=mmin_AE --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=cmumosei_AETemp

## Step4: train CRA model (change dataset_mode and suffix)
python -u train_miss.py --mask_rate=0.2 --dataset_mode=cmumosi_miss     --n_blocks=2 --model=mmin_CRA --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.8 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=cmumosi_CRATemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapfour_miss --n_blocks=2 --model=mmin_CRA --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.8 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapfour_CRATemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=iemocapsix_miss  --n_blocks=2 --model=mmin_CRA --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.8 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=iemocapsix_CRATemp
python -u train_miss.py --mask_rate=0.2 --dataset_mode=cmumosei_miss    --n_blocks=2 --model=mmin_CRA --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=0 --input_dim_a=512 --embd_size_a=128 --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.8 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=80 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin --suffix=cmumosei_CRATemp


################################################
######### Train cmpnet (tf115env37) #########
################################################
cd baseline-cpmnet

## change feature format
python change_format.py change_feat_format_cmumosi
python change_format.py change_feat_format_iemocapfour
python change_format.py change_feat_format_iemocapsix
python change_format.py change_feat_format_cmumosei

## training model
python test_lianzheng.py --dataset='cmumosi'     --missing-rate=0.2 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=1
python test_lianzheng.py --dataset='iemocapfour' --missing-rate=0.2 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=1
python test_lianzheng.py --dataset='iemocapsix'  --missing-rate=0.2 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=1
python test_lianzheng.py --dataset='cmumosei'    --missing-rate=0.2 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=1


###############################################
######### Train CCA/DCCA/DCCAE ################
###############################################
# training with cpmnet-generated data format
cd baseline-cca

python cca.py   --dataset='cmumosi' --missing-rate=0.2 --n-components=2
python dcca.py  --dataset='cmumosi' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset='cmumosi' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2

python cca.py   --dataset='iemocapfour' --missing-rate=0.2 --n-components=2
python dcca.py  --dataset='iemocapfour' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset='iemocapfour' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2

python cca.py   --dataset='iemocapsix' --missing-rate=0.2 --n-components=2
python dcca.py  --dataset='iemocapsix' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset='iemocapsix' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2

python cca.py   --dataset='cmumosei' --missing-rate=0.2 --n-components=2
python dcca.py  --dataset='cmumosei' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset='cmumosei' --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2


###########################################
######### whole parameter tuning ##########
###########################################
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
