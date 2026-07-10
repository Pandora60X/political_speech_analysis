import re
import nltk
import spacy

STOPWORDS = nltk.corpus.stopwords.words('english')

def clean_text(text, remove_punctuation=True, lowercase=True, remove_digits=True):
   if remove_punctuation:
      text = re.sub(r'\W+', ' ', text)
   
   if lowercase:
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

def preprocess(text):
   text = clean_text(text)
   tokens = tokenize(text)
   clean_tokens = remove_stopwords(tokens)
   return clean_tokens

def spacy_tokenize(text):
   nlp = spacy.load("en_core_web_sm")
   doc = nlp(text)
   return doc
