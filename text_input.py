from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty, ListProperty
from kivymd.theming import ThemableBehavior
from kivymd.uix.list import OneLineIconListItem, MDList
from kivymd.uix.card import MDCard
from kivy.metrics import dp
from kivy.core.window import Window
import tensorflow as tf
import numpy as np
import os
import time
import json
import pandas


with open('char2idx.json', 'r') as f:
    char2idx = json.load(f)


idx2char = []
with open("idx2char.txt") as f:
    for line in f:
        idx2char.append(line)

idx2char = list(map(lambda x:x.strip("\n"),idx2char))
idx2char.insert(0, "\n")
idx2char.remove("")
idx2char.remove("")

class UI(Screen):
    pass

class Result(Screen):
     pass



class MyApp(MDApp):

    # layout
    def build(self):
        sm = ScreenManager()
        sm.add_widget(UI(name='first'))
        sm.add_widget(Result(name='second'))
        return sm

    # button click function
    '''def buttonClicked(self,btn):
        self.lbl1.text = "Generated text: " + self.txt1.text'''
    '''def get_temp(self):
        temperature =  self.root.ids.temp.text
        return self.temperature
    def get_num(self):
        num_generate =  self.root.ids.num.text
        return self.num_generate'''
    def get_word(self):
        start_str =  self.root.ids.word.text
        return print(start_str)

    # generate text using DL model
    ## adding model, start_string, num_generate, temperature
    def generate_text(self, num_generate = 500, temperature = 1, start_string = 'Smoke'):

      model = tf.keras.models.load_model('good_model_100_epochs.h5', compile=False)
      '''num_generate = int(self.root.ids.num)
      temperature = int(self.root.ids.temp)
      start_string = self.root.ids.word'''
      # Evaluation step (generating text using the learned model)

      # Number of characters to generate
      num_generate = num_generate

      # Converting our start string to numbers (vectorizing)
      input_eval = [char2idx[s] for s in start_string]
      input_eval = tf.expand_dims(input_eval, 0)

      # Empty string to store our results
      text_generated = []

      # Low temperatures results in more predictable text.
      # Higher temperatures results in more surprising text.
      # Experiment to find the best setting.
      # You can try experimenting with values between 0 and 2 but values lower than 0.6 are very likely to just copy existing lyrics
      # and values higher than 1.5 are going to be random gibberish (remember to use . not ,)
      temperature = temperature

      # Here batch size == 1
      model.reset_states()
      for i in range(num_generate):
          predictions = model(input_eval)
          # remove the batch dimension
          predictions = tf.squeeze(predictions, 0)

          # using a categorical distribution to predict the character returned by the model
          predictions = predictions / temperature
          predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

          # We pass the predicted character as the next input to the model
          # along with the previous hidden state
          input_eval = tf.expand_dims([predicted_id], 0)

          text_generated.append(idx2char[predicted_id])

      result = start_string + ''.join(text_generated)

      return print(result)
      return result

    def print_lyrics(self):
        return print(result)

# run app
if __name__ == "__main__":
    MyApp().run()
