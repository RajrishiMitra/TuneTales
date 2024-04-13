import speech_recognition as sr

def speech_to_text():
    # InitiSalize recognizer
    recognizer = sr.Recognizer()

    # Capture audio from the microphone
    with sr.Microphone() as source:
        #print("Listening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Listen for user input
        audio = recognizer.listen(source)

        #print("Recognizing...")

    try:
        # Use Google Speech Recognition
        text = recognizer.recognize_google(audio)
        #print("You said:", text)
        return text

    except sr.UnknownValueError:
        print("Could not understand audio")
        return None

    except sr.RequestError as e:
        print("Error occurred during recognition: {0}".format(e))
        return None




