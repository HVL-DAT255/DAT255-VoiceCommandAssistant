import pvporcupine
import pyaudio
import struct
import os
from dotenv import load_dotenv

# Last inn .env-variabler
load_dotenv()


class WakeWordDetector:
    def __init__(self):
        self.porcupine = pvporcupine.create(
            access_key=os.getenv("PICOVOICE_ACCESS_KEY"),  # Hent API-nøkkel fra .env
            keyword_paths=[os.path.join(os.path.dirname(__file__), "porcupine_model", "hey_siri.ppn")]
        )

        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

    def listen(self):
        print("🎤 Lytter etter 'Hey Siri'... ")
        try:
            while True:
                pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                keyword_index = self.porcupine.process(pcm)

                if keyword_index >= 0:
                    print("🎉 Oppdaget 'Hey Siri'!")
                    return True

        except KeyboardInterrupt:
            print("🛑 Avslutter wake word detection.")
        finally:
            self.audio_stream.close()
            self.pa.terminate()
            self.porcupine.delete()

if __name__ == "__main__":
    detector = WakeWordDetector()
    detector.listen()