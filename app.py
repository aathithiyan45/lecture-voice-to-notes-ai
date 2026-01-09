import speech_recognition as sr

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        
    try:
        text = recognizer.recognize_google(audio)
        return text
    except:
        return "Sorry, could not understand the audio"

if __name__ == "__main__":
    text = speech_to_text("audio.wav")
    print("Converted Text:")
    print(text)
