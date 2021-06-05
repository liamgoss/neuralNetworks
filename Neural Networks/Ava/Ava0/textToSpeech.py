# pyttsx3
'''
import pyttsx3

engine = pyttsx3.init()

engine.say("Testing 1 2 3")
engine.runAndWait()

# Change rate
rate = engine.getProperty("rate")
print(rate)
engine.setProperty("rate", 175)
engine.say("Testing 3 2 1")
engine.runAndWait()

# Change volume
volume = engine.getProperty("volume")
print(volume)
engine.setProperty("volume", 1.0)  # between 0 and 1

engine.say("Testing testing")
engine.runAndWait()

# Change voice
voices = engine.getProperty("voices")
# 0 for male, 1 for female
engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 150)
myText = "Welcome, my name is Winston! Wonderful fucking weather we are having, eh?"
engine.say(myText)
engine.runAndWait()
'''

# gTTS (Google's TTS)
from gtts import gTTS
from playsound import playsound
import os
from tkinter import *
# https://www.tutorialsteacher.com/python/create-gui-using-tkinter-python
'''
Accent      |       Language Code       |       Top-Level Domain
Australia               en                          com.au
UK                      en                          co.uk
US                      en                          com
Canada                  en                          ca
India                   en                          co.in
Ireland                 en                          ie
South Africa            en                          co.za
'''
language = "en"
tld = "co.uk"

window = Tk()
window.title("Ava")
window.geometry("300x200+50+150")


def generateSpeech(textToSpeak):
    if textToSpeak != "" and not textToSpeak.isspace() and textToSpeak != "Enter text to be spoken":
        #textToSpeak = "Hello my name is Ayvah and I am your digital companion!"
        #textToSpeak = "Okay Arya, which pronunciation is best? Ava, or, Ayvah"
        filename = "tmp.mp3"
        tmpObj = gTTS(text=textToSpeak, lang=language, tld=tld, slow=False)
        tmpObj.save(filename)
        playsound(filename)
        os.remove(filename)

class EntryWithPlaceholder(Entry):
    def __init__(self, master=None, placeholder="PLACEHOLDER", color='grey'):
        super().__init__(master)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']

        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)

        self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

    def foc_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self['fg'] = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()

class MainWindow:
    def __init__(self, win):
        self.txtField = EntryWithPlaceholder(win, "Enter text to be spoken")
        self.speakBtn = Button(window, text="Speak Text", command=lambda: generateSpeech(self.txtField.get()), fg="blue")
        self.speakBtn.place(relx=0.5, rely=0.5, anchor="center")  # Anchor can be center, NW, SE, etc
        self.txtField.place(in_=self.speakBtn, relx=-0.5, rely=-2)  # in_ means relative to another widget


#txtField = EntryWithPlaceholder(window, "Enter text to be spoken")
# Button Properties: text, bg, command, fg, font, image
#speakBtn = Button(window, text="Speak Text", command=lambda: generateSpeech(txtField.get()), fg="blue")
#speakBtn.place(relx=0.5, rely=0.5, anchor="center")  # Anchor can be center, NW, SE, etc


#txtField.place(in_=speakBtn, relx=-0.5, rely=-2)  # in_ means relative to another widget

#lbl = Label(window, text="Click the button for me to introduce myself!", font=("Helvetica", 16), wraplength=300, justify="center")
#lbl.pack()

win = MainWindow(window)
window.mainloop()



