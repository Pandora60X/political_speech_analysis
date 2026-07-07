import re
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from wordcloud import WordCloud
import matplotlib.pyplot as plt


'''Processing Steps:
- clean
- tokenize
- remove stopwords
'''

stopwords = nltk.corpus.stopwords.words('english')



def process_speech(text):
   #Clean text by removing punctuation and numbers
   text=text.lower()
   text = re.sub(r'\W', ' ', text)
   text = re.sub(r'\d', ' ', text)
   
   #Tokenize text
   speech_tokens = nltk.tokenize.word_tokenize(text)

   #Remove stopwords
   stopword_set=set(nltk.corpus.stopwords.words('english'))

   words = [word for word in speech_tokens if word not in stopword_set]

   return ' '.join(words)

def main():
   with open('address_to_the_nation_on_iran.txt', 'r', encoding="utf-8") as file:
      raw_speech=file.read()

   speech_string = process_speech(raw_speech)

   wc = WordCloud().generate(''.join(speech_string))
   plt.imshow(wc)
   plt.show()

if __name__ == "__main__":
   main()


