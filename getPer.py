from dtw import dtw
import os
import librosa
import numpy as np

def load_audio(filename):
    y, sr =librosa.load(filename, sr=None)
    return y, sr

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc
 
def calculate_similarity(mfcc1, mfcc2):
    # DTW 알고리즘을 사용하여 두 MFCC 시퀀스 간의 거리 계산
    d, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    # 거리 값이 클수록 두 오디오가 다르고, 작을수록 유사함
    similarity = 1 / (1 + d)
    return similarity

if __name__ == "__main__":
    # 두 개의 오디오 파일 로드
    voice_folder = "voicefolder"
    original_voice_folder = "originalVoice"
    voice_file = "minsic.wav"
    original_voice_file = "minsic.wav"

    voice_path = os.path.join(voice_folder, voice_file)
    original_voice_path = os.path.join(original_voice_folder, original_voice_file)

    voice, sr_voice = load_audio(voice_path)
    original_voice, sr_original_voice = load_audio(original_voice_path)

    # MFCC 추출
    mfcc_voice = extract_mfcc(voice, sr_voice)
    mfcc_original_voice = extract_mfcc(original_voice, sr_original_voice)

    # 유사도 계산
    similarity = calculate_similarity(mfcc_voice, mfcc_original_voice)
    print("Similarity between the two voices:", similarity)