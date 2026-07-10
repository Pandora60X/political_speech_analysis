import re
import nltk

STOPWORDS = nltk.corpus.stopwords.words('english')

def clean_text(text, remove_punctuation=True, remove_caps=True, remove_digits=True):
   if remove_punctuation:
      text = re.sub(r'\W+', ' ', text)
   
   if remove_caps:
      text = text.lower()

   if remove_digits:
      text = re.sub(r'\d', ' ', text)

   text = re.sub(r'\s+', ' ', text)
   text = text.strip()

   return text

def tokenize(text):
   return nltk.tokenize.word_tokenize(text)

def remove_stopwords(tokens):
   clean_tokens = [word for word in tokens if word not in STOPWORDS]
   return clean_tokens
