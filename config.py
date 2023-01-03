# *_*coding:utf-8 *_*
import os
import sys
import socket

## gain linux ip
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('10.0.0.1',8080))
        ip= s.getsockname()[0]
    finally:
        s.close()
    return ip

############ For LINUX ##############
# path
DATA_DIR = {
	'CMUMOSI': '/share/home/lianzheng/gcnet-master/dataset/CMUMOSI',   # for nlpr
	'CMUMOSEI': '/share/home/lianzheng/gcnet-master/dataset/CMUMOSEI',# for nlpr
	'IEMOCAPSix': '/share/home/lianzheng/gcnet-master/dataset/IEMOCAP', # for nlpr
	'IEMOCAPFour': '/share/home/lianzheng/gcnet-master/dataset/IEMOCAP', # for nlpr
}
PATH_TO_RAW_AUDIO = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subaudio'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
}
PATH_TO_RAW_FACE = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subvideofaces'), # without openfac
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideofaces'),
}
PATH_TO_TRANSCRIPTIONS = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription.csv'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription.csv'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'transcription.csv'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'features'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'features'),
}
PATH_TO_LABEL = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}

# pre-trained models, including supervised and unsupervised
PATH_TO_PRETRAINED_MODELS = '/share/home/lianzheng/tools'
PATH_TO_OPENSMILE = '/share/home/lianzheng/tools/opensmile-2.3.0/'
PATH_TO_FFMPEG = '/share/home/lianzheng/tools/ffmpeg-4.4.1-i686-static/ffmpeg'

# dir
SAVED_ROOT = os.path.join('../saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')



############ For Windows ##############
DATA_DIR_Win = {
	'CMUMOSI': 'E:\\Dataset\\CMU-MOSI\\Raw',
	'CMUMOSEI1': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
	'CMUMOSEI2': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
	'CMUMOSEI3': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
	'CMUMOSEI4': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
	'CMUMOSEI5': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
}

PATH_TO_RAW_FACE_Win = {
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'Video\\Segmented'),
	'CMUMOSEI1': os.path.join(DATA_DIR_Win['CMUMOSEI1'], 'subvideo1'),
	'CMUMOSEI2': os.path.join(DATA_DIR_Win['CMUMOSEI2'], 'subvideo2'),
	'CMUMOSEI3': os.path.join(DATA_DIR_Win['CMUMOSEI3'], 'subvideo3'),
	'CMUMOSEI4': os.path.join(DATA_DIR_Win['CMUMOSEI4'], 'subvideo4'),
	'CMUMOSEI5': os.path.join(DATA_DIR_Win['CMUMOSEI5'], 'subvideo5'),
}

PATH_TO_FEATURES_Win = {
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'features'),
	'CMUMOSEI1': os.path.join(DATA_DIR_Win['CMUMOSEI1'], 'features'),
	'CMUMOSEI2': os.path.join(DATA_DIR_Win['CMUMOSEI2'], 'features'),
	'CMUMOSEI3': os.path.join(DATA_DIR_Win['CMUMOSEI3'], 'features'),
	'CMUMOSEI4': os.path.join(DATA_DIR_Win['CMUMOSEI4'], 'features'),
	'CMUMOSEI5': os.path.join(DATA_DIR_Win['CMUMOSEI5'], 'features'),
}

PATH_TO_OPENFACE_Win = "H:\\desktop\\Multimedia-Transformer\\gcnet-master\\OpenFace_2.2.0_win_x64\\OpenFace_2.2.0_win_x64"
PATH_TO_FFMPEG_Win = "H:\\desktop\\Multimedia-Transformer\\tools\\ffmpeg-3.4.1-win32-static\\bin\\ffmpeg"

