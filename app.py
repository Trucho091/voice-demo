import gradio as gr
import numpy as np
import librosa
import tensorflow as tf

model = tf.keras.models.load_model("gender_voice_best.keras")

SR = 16000
DURATION = 3.0
N_SAMPLES = int(SR * DURATION)

IMG_H = 128
IMG_W = 128
N_MELS = 128
N_FFT = 1024
HOP = 256

def preprocess(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)

    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    y = y / (np.max(np.abs(y)) + 1e-8)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP
    )

    log_mel = librosa.power_to_db(mel)
    log_mel = tf.image.resize(log_mel[..., np.newaxis], (IMG_H, IMG_W))

    return np.array(log_mel)

def predict(audio):
    x = preprocess(audio)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0]

    return {
        "Female": float(pred[0]),
        "Male": float(pred[1]),
        "Noise": float(pred[2])
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Label(),
    title="🎤 Gender Voice Classifier"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
