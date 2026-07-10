import re
import nltk

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
   cleaned_text = clean_text(text)
   tokens = tokenize(cleaned_text)
   clean_tokens = remove_stopwords(tokens)
   return clean_tokens