# emotion-in-voice

Classifies the emotion in a speech clip (e.g. happy, sad, angry, calm) using deep learning.

## Demo

Demo Link: https://emotion-in-voice-new.streamlit.app/

## How it works

- **Dataset:** RAVDESS (speech audio from 24 actors, emotion label encoded in each filename)
- **Model:** [Wav2Vec2-base fine-tuned on RAVDESS / CNN on spectrograms. Say which one you actually used]
- **App:** `app.py` loads the trained model and predicts the emotion from an uploaded audio file using [Streamlit / Gradio / Flask]

## Run it locally

1. Clone the repo:
   `git clone https://github.com/[your-username]/emotion-in-voice.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start the app:
   `[streamlit run app.py / python app.py]`

## Tech stack

Python, PyTorch, [Hugging Face Transformers, librosa, etc. Match your requirements.txt]

## Limitations

- Trained on acted speech (RAVDESS), so it may perform worse on real-world, noisy audio
- English only

## Author

Talha, BS Data Science student
Linkden: https://www.linkedin.com/in/talhabutt7810/
