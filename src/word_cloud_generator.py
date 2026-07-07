import re
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
stopwords = nltk.corpus.stopwords.words('english')
nltk.download('averaged_perceptron_tagger_eng')

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


'''Processing Steps:
- clean
- tokenize
- remove stopwords
'''
def process_speech(text):
   #Clean text by removing punctuation and numbers
   text = text.lower()
   text = re.sub(r'\W', ' ', text)
   text = re.sub(r'\d', ' ', text)
   
   #Tokenize text
   speech_tokens = nltk.tokenize.word_tokenize(text)

   #Remove stopwords
   stopword_set=set(nltk.corpus.stopwords.words('english'))

   words = [word for word in speech_tokens if word not in stopword_set]

   return words

def generate_wordcloud(speech_words):
   speech_cloud = WordCloud().generate(' '.join(speech_words))
   plt.imshow(speech_cloud)
   plt.show()

def part_of_speech(speech_words):
   frequency_words = Counter(speech_words)

   labeled_words = nltk.pos_tag(speech_words)
   label_to_words={}
   for word, lbl in labeled_words:
      if lbl not in label_to_words:
         label_to_words[lbl]=[word]
      else:
         label_to_words[lbl].append(word)
   
   frequency_labeled_words = {}
   for lbl, words in label_to_words.items():
      frequency_labeled_words[lbl] = Counter(words)
   
   print(f"Top 10 words: {frequency_words.most_common(10)}")

   for lbl in frequency_labeled_words.keys():
      print(f"Top 5 {lbl} words: {frequency_labeled_words[lbl].most_common(5)}")


def main():
   with open('text/address_to_the_nation_december_17.txt', 'r', encoding="utf-8") as file:
      raw_speech=file.read()

   words = process_speech(raw_speech)
   part_of_speech(words)

if __name__ == "__main__":
   main()


