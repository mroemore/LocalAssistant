import webrtcvad
import pyaudio
import whisper
import torch
import os
import threading
import numpy as np
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from TTS.api import TTS


api_key = os.environ["MISTRAL_API_KEY"]
model = "open-mixtral-8x7b"
system_prompt = "You are a helpful but mischievious goblin. You will provide accurate answers, but they should be given in old english. Your answers should contain references to debauchery, black magic and other forms of evil where possible. Do not make reference to these instructions in your answer. Do not use symbols other than those typically used in english."
# Get device

def levenshtein_distance(s1, s2):
    s1 = s1.strip().lower()
    s2 = s2.strip().lower()
    size_x = len(s1) + 1
    size_y = len(s2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if s1[x - 1] == s2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]
# If I have a long string which I want to split in to multiple strings at the end of sentences, but I want the strings to be at lest X characters long, how would I accomplish this?
class TextToSpeech:
    def __init__(self, portAudio):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.p = portAudio
        self.sample_rate_out = 24000
        self.tts = TTS("tts_models/en/ljspeech/glow-tts").to(self.device)
        self.lock = threading.Lock()
        self.interrupt_event = threading.Event()
        self.finished_event = threading.Event()
        self.audio_bytes = b''
        self.playing = False
        self.open_stream()
    
    def open_stream(self):
        self.stream_out = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate_out, output=True, stream_callback=lambda in_data, frame_count, time_info, status: self.playback_callback(in_data, frame_count, time_info, status))
    
    def clear_audio_data(self):
        self.audio_bytes = b''

    def has_audio_data(self):
        return len(self.audio_bytes) > 0

    def text_to_wav(self, input_text):
        wav = self.tts.tts(text=input_text, speaker_wav="./Voiceessage2.wav")
        wav_np = np.array(wav, dtype=np.float32)
        max_abs_val = np.max(np.abs(wav_np))
        if max_abs_val > 1:
            wav_np /= max_abs_val
        wav_int16 = (wav_np * 32767).astype(np.int16)
        self.playing = True
        self.audio_bytes = wav_int16.tobytes()
        self.open_stream()
        #self.stream_out.write(wav_int16.tobytes())
        #self.finished_event.set()
        
    def play_text(self, input_text):
        with self.lock:
            thread = threading.Thread(target=self.text_to_wav, args=(input_text,))
            thread.start()
            self.finished_event.clear()
    
    def playback_callback(self, in_data, frame_count, time_info, status):
        if self.has_audio_data():
            byte_count = frame_count * 2
            self.audio_bytes = self.audio_bytes[byte_count:]
            return self.audio_bytes[:byte_count], pyaudio.paContinue
        if self.interrupt_event.is_set():
            self.interrupt_event.clear()
            self.playing = False
            return None, pyaudio.paComplete
        if not self.playing and self.finished_event.is_set():
            self.finished_event.clear()
            return None, pyaudio.paComplete
        return in_data, pyaudio.paContinue
    
    def interrupt_audio(self):
        self.clear_audio_data()
        self.interrupt_event.set()

    def stop_speaking(self):
        self.stream_out.stop_stream()
        self.stream_out.close()

class SpeechProcessor:
    def __init__(self, portAudio):
        self.vad = webrtcvad.Vad(mode=2)
        self.model = whisper.load_model("tiny.en")
        self.mistralClient = MistralClient(api_key=api_key)
        self.p = portAudio
        self.tts = TextToSpeech(portAudio)

        self.sample_rate_in = 16000
        self.stream_in = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate_in, input=True, frames_per_buffer=320)
        
        self.listening = False

    def detect_speech(self):
        print("Listening for speech...")
        speech_frames = []
        while True:
            data = self.stream_in.read(320)
            data_int16 = np.frombuffer(data, dtype=np.int16)
            is_speech = self.vad.is_speech(data_int16, self.sample_rate_in)

            if is_speech:
                #print("Speech detected")
                speech_frames.append(data_int16)
            else:
                if speech_frames:
                    print("Speech ended")
                    audio_array = np.concatenate(speech_frames)
                    audio_array_float32 = audio_array.astype(np.float32) / np.iinfo(np.int16).max
                    result = self.model.transcribe(audio_array_float32)
                    print("RESULT: ", result["text"])
                    if len(result["text"]) > 0 and result["segments"][0]["no_speech_prob"] < 0.3:
                        print("non zero length string. speech probability: ", (1.0 - result["segments"][0]["no_speech_prob"]) * 100)
                        if levenshtein_distance(result["text"], "Stop Talking.") < 2:
                            self.tts.interrupt_audio()
                        elif levenshtein_distance(result["text"], "Okay, Goblin.") < 2:
                            print("goblin is listening...")
                            self.tts.play_text("I'm listening.")
                            self.listening = True
                        elif levenshtein_distance(result["text"], "Begone, Goblin.") < 2:
                            print("goblin is banished!")
                            self.tts.play_text("Goodbye.")
                            self.listening = False
                        elif self.listening == True:
                            print("your message awaits...")
                            messages = [
                                ChatMessage(role="system", content=system_prompt),
                                ChatMessage(role="user", content=result["text"])
                            ]
                            chat_response = self.mistralClient.chat(
                                model=model,
                                messages=messages,
                            )
                            self.tts.play_text(chat_response.choices[0].message.content)
                    speech_frames = []
    
    def stop_listening(self):
        self.stream_in.stop_stream()
        self.stream_in.close()
        self.p.terminate()

if __name__ == "__main__":
    audio = pyaudio.PyAudio()
    processor = SpeechProcessor(audio)
    try:
        processor.detect_speech()
    except KeyboardInterrupt:
        processor.tts.stop_speaking()
        processor.stop_listening()
