from preprocessing import preprocess

import re
import nltk

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


def generate_wordcloud(tokens):
   speech_cloud = WordCloud().generate(' '.join(tokens))
   plt.imshow(speech_cloud)
   plt.show()


'''Print
- overall highest word frequency
- highest word frequency by part of speech'''
def part_of_speech(speech_words):
   frequency_words = Counter(speech_words)

   labeled_words = nltk.pos_tag(speech_words)

   #Dictionary with labels as keys and list of words as values
   label_to_words={'NOUN':[], 'VERB':[], 'ADJECTIVE':[], 'ADVERB':[]}
   for word, lbl in labeled_words:
      if lbl.startswith("NN"):
         label_to_words['NOUN'].append(word)
      elif lbl.startswith("VB"):
         label_to_words['VERB'].append(word)
      elif lbl.startswith("JJ"):
         label_to_words['ADJECTIVE'].append(word)
      elif lbl.startswith("RB"):
         label_to_words['ADVERB'].append(word)
   
   #Dictionary with labels as keys and Counter object of words as values
   frequency_labeled_words = {}
   for lbl, words in label_to_words.items():
      frequency_labeled_words[lbl] = Counter(words)
   
   print(f"Top 10 words: {frequency_words.most_common(10)}")

   for lbl in frequency_labeled_words.keys():
      print(f"Top 5 {lbl} words: {frequency_labeled_words[lbl].most_common(5)}")


def main():
   with open('text/trump_state_of_union_2_24_26.txt', 'r', encoding="utf-8") as file:
      raw_speech=file.read()

   tokens = preprocess(raw_speech)
   part_of_speech(tokens)

if __name__ == "__main__":
   main()


