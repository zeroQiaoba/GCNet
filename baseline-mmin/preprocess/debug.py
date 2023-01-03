import sys
import os

def show_wdseg(utt_id):
    root = '/data3/lrc/IEMOCAP_full_release'
    word_info_dir = os.path.join(root, 'Session{}/sentences/ForcedAlignment/{}')
    session_id = int(utt_id[4])
    dialog_id = '_'.join(utt_id.split('_')[:-1])
    word_info_path = os.path.join(word_info_dir.format(session_id, dialog_id), utt_id + '.wdseg')
    print(f'{utt_id} wdset info:')
    print(open(word_info_path, 'r').read())

def show_sentence(utt_id):
    root = '/data3/lrc/IEMOCAP_full_release'
    sentence_dir = os.path.join(root, 'Session{}/dialog/transcriptions/{}.txt')
    session_id = int(utt_id[4])
    dialog_id = '_'.join(utt_id.split('_')[:-1])
    transcript_path = sentence_dir.format(session_id, dialog_id)
    print(f'{utt_id} transcripts:')
    for line in open(transcript_path).readlines():
        if line.startswith(utt_id):
            print(line)
            break

if __name__ == '__main__':
    utt_id = sys.argv[1]
    show_wdseg(utt_id)
    show_sentence(utt_id)
