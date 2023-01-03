python extract_handcrafted_feature.py --dataset='CHEAVD' --feature_extractor='pyAudio' --feature_set='pyAudio' --feature_level='UTTERANCE'
python extract_handcrafted_feature.py --dataset='CHEAVD' --feature_extractor='opensmile' --feature_set='IS09' --feature_level='UTTERANCE'
python extract_handcrafted_feature.py --dataset='CHEAVD' --feature_extractor='opensmile' --feature_set='IS10' --feature_level='UTTERANCE'
python extract_handcrafted_feature.py --dataset='CHEAVD' --feature_extractor='opensmile' --feature_set='IS13' --feature_level='UTTERANCE'
python extract_handcrafted_feature.py --dataset='CHEAVD' --feature_extractor='opensmile' --feature_set='eGeMAPS' --feature_level='UTTERANCE'
python extract_handcrafted_feature.py --dataset='CHEAVD' --feature_extractor='Librosa' --feature_set='mel_spec' --feature_level='UTTERANCE'
python extract_handcrafted_feature.py --dataset='CHEAVD' --feature_extractor='Librosa' --feature_set='mfcc' --feature_level='UTTERANCE'
python extract_wav2vec_embedding.py --dataset='CHEAVD' --feature_level='UTTERANCE' --gpu=0
python extract_wav2vec2_embedding.py --dataset='CHEAVD' --model_name='wav2vec2-base' --feature_level='UTTERANCE' --gpu=0
python extract_wav2vec2_embedding.py --dataset='CHEAVD' --model_name='wav2vec2-base-960h' --feature_level='UTTERANCE' --gpu=0
python extract_wav2vec2_embedding.py --dataset='CHEAVD' --model_name='wav2vec2-large-960h' --feature_level='UTTERANCE' --gpu=0
python extract_panns_embedding.py --dataset='CHEAVD' --feature_level='UTTERANCE' --gpu=0
python extract_vggish_embedding.py --dataset='CHEAVD' --feature_level='UTTERANCE' --gpu=0