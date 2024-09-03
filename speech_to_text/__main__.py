import whisper
import numpy as np

CHUNK_LIM = 480000

model = whisper.load_model("medium")

audios = []
audio = whisper.load_audio("../audios/example.mp3")  # insert the path to your audiofile

if len(audio) <= CHUNK_LIM:
    audios.append(audio)
else:
    for i in range(0, len(audio), CHUNK_LIM):
        chunk = audio[i:i + CHUNK_LIM]
        chunk_index = len(chunk)
        if chunk_index < CHUNK_LIM:
            padding = [0] * (CHUNK_LIM - chunk_index)
            array1 = np.array(chunk)
            array2 = np.array(padding)
            concat = np.concatenate((array1, array2))
            chunk = concat.astype(np.float32)
        audios.append(chunk)

results = ""

for chunk in audios:
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(chunk).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    results += result.text

print(results)
f = open("../your.txt", "a")  # insert the filename here
f.write(results)
f.close()
