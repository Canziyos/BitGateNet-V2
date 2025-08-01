import sounddevice as sd
import soundfile as sf

samplerate = 16000 
duration = 1.0 
filename = "test_audio.wav"

print(">>> Recording for 1 second...")
audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
sd.wait() 

sf.write(filename, audio, samplerate)
print(f"Saved to {filename}")
