import streamlit as st
import librosa
import torch
import numpy as np
import io
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"

st.set_page_config(page_title="Voice Emotion Detector", page_icon="🎙️")


@st.cache_resource
def load_model():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    return feature_extractor, model


def predict_emotion(audio_bytes, feature_extractor, model):
    # librosa can load directly from a file-like object
    audio, rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    inputs = feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        outputs = model(inputs.input_values)
        probs = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)

    probs = probs.squeeze().numpy()
    predicted_id = int(np.argmax(probs))
    predicted_label = model.config.id2label[predicted_id]

    # full distribution, sorted high -> low
    label_probs = {
        model.config.id2label[i]: float(probs[i]) for i in range(len(probs))
    }
    label_probs = dict(sorted(label_probs.items(), key=lambda x: x[1], reverse=True))

    return predicted_label, label_probs


st.title("🎙️ Voice Emotion Detector")
st.write("Record your voice and the model will predict the emotion.")

feature_extractor, model = load_model()

audio_value = st.audio_input("Click the mic and record your voice")

if audio_value is not None:
    st.audio(audio_value)

    if st.button("Predict Emotion"):
        with st.spinner("Analyzing..."):
            audio_bytes = audio_value.read()
            predicted_label, label_probs = predict_emotion(
                audio_bytes, feature_extractor, model
            )

        st.success(f"Predicted Emotion: **{predicted_label.upper()}**")

        st.subheader("Confidence scores")
        st.bar_chart(label_probs)