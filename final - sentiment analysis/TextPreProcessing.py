import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk


# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# using spacy
import spacy
nlp = spacy.load('en_core_web_sm',  disable=["parser", "ner"])

# Preprocess text fields
def preprocess(document):
  # 1. Change the document to lower case
  document = document.lower()

  # 2. Remove punctuation and words containing numbers
  document = re.sub("[^\sA-z]","",document)

  # 3. Tokenize the words
  words = word_tokenize(document)

  # 4. Remove the stop words
  words = [word for word in words if word not in stopwords.words("english")]

  # 5. Remove words with 0 or 1 letter
  words = [w for w in words if len(w) > 1]

  # 6. join
  document = " ".join(words)
  return(document)



# Applying Lemmatization
def lemmatize_text(text):
    sent = []
    doc = nlp(text)
    for token in doc:
        sent.append(token.lemma_)
    return " ".join(sent)


#function to collect the n-gram frequency of words
def get_top_n_ngram( corpus, n_gram_range ,n=None):
    vec = CountVectorizer(ngram_range=(n_gram_range, n_gram_range), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    print("--1",sum_words)
    for word, idx in vec.vocabulary_.items():
        break
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]