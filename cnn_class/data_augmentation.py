from cremadata import CremaSoundDataset
import matplotlib.pyplot as plt
import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt

def add_white_noise(signal, noise_factor=0.5):
    noise = np.random.randn(*signal.shape) * (signal.std() * noise_factor)
    augmented_signal = signal + noise
    return augmented_signal





if __name__ == "__main__":

    BATCH_SIZE = 16
    EPOCHS = 150
    LEARNING_RATE = 1e-3
    ANNOTATIONS_FILE = "SentenceFilenames.csv"
    AUDIO_DIR = "AudioWAV"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = SAMPLE_RATE*4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device: ", device)



    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = CremaSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    aug = add_white_noise(usd[0][0].numpy(), 0.1)
    aug = aug.squeeze()
    aug = aug.reshape(-1)
    sf.write("augmented.wav", aug, SAMPLE_RATE)
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    librosa.display.waveshow(usd[0][0].numpy(), sr=SAMPLE_RATE, ax=ax[0])
    ax[0].set(title="Original")
    librosa.display.waveshow(aug, sr=SAMPLE_RATE, ax=ax[1])
    ax[1].set(title="Augmented")
    plt.show()