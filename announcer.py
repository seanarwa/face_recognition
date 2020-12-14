import pyttsx3

class TTS:

    engine = None

    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)

    def say(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

def say(text):
    tts = TTS()
    tts.say(text)
    del(tts)
