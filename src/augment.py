import numpy as np
import librosa
import random

SAMPLE_RATE = 16_000

def add_white_noise(y, snr_db=20):
    # Compute RMS of clean signal
    rms_clean = np.sqrt(np.mean(y ** 2))
    # Compute RMS of noise for desired SNR
    snr_linear = 10 ** (snr_db / 20)
    rms_noise = rms_clean / snr_linear
    noise = np.random.normal(scale=rms_noise, size=len(y))
    return y + noise

def time_shift(y, shift_max=0.2):
    """Shift by up to ±shift_max seconds (wrap mode)."""
    max_s = int(shift_max * SAMPLE_RATE)
    shift = random.randint(-max_s, max_s)
    return np.roll(y, shift)

def pitch_shift(y, n_steps_max=2):
    n_steps = random.uniform(-n_steps_max, n_steps_max)
    return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)

# single helper that chains ops with 50 % probability each
def augment_waveform(y):
    if random.random() < .5:
        y = add_white_noise(y, snr_db=random.uniform(10, 30))
    if random.random() < .5:
        y = time_shift(y)
    if random.random() < .5:
        y = pitch_shift(y)
    return y
