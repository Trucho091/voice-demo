import os
import gradio as gr
import numpy as np
import librosa
import tensorflow as tf

print("Starting app...")
print("Loading model...")

model = tf.keras.models.load_model("crnn_best.keras", compile=False)

print("Model loaded successfully.")

SR = 16000
DURATION = 3.0
N_SAMPLES = int(SR * DURATION)

IMG_H = 128
IMG_W = 128
N_MELS = 128
N_FFT = 1024
HOP = 256


def preprocess(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    y = y / (np.max(np.abs(y)) + 1e-8)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = tf.image.resize(log_mel[..., np.newaxis], (IMG_H, IMG_W))
    log_mel = np.array(log_mel, dtype=np.float32)

    return log_mel


def predict(audio):
    if audio is None:
        return {"Female": 0.0, "Male": 0.0, "Noise": 0.0}

    x = preprocess(audio)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)[0]

    return {
        "Female": float(pred[0]),
        "Male": float(pred[1]),
        "Noise": float(pred[2])
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload audio"),
    outputs=gr.Label(label="Prediction"),
    title="🎤 Gender Voice Classifier",
    description="Upload audio để phân loại: Female / Male / Noise"
)

print("Launching Gradio...")

port = int(os.environ.get("PORT", 7860))
demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    show_error=True
)
