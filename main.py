import os
import pyaudio
import librosa
import wave

import numpy as np
from dtw import dtw

from datetime import datetime




def record_audio(filename, duration=5, sr=44100):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=sr, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []
    for i in range(0, int(sr / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 저장될 폴더 경로
    folder = "voicefolder"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 파일 경로 설정
    filepath= os.path.join(folder,filename)

    wf = wave.open(filepath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(sr)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Audio saved as", filepath)

def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr

if __name__ == "__main__":
    # 파일생성 이름 바꾸기
    now = datetime.now()
    now = str(now)
    now = now.split(".")[0]
    now = now.replace("-","").replace(" ","_").replace(":","")
    # filename = "recorded_audio.wav"
    filename = now+'.wav'
    record_audio(filename)
    filepath = os.path.join("voicefolder", filename)
    y, sr = load_audio(filename)
    print("Loaded audio:", y.shape, sr)