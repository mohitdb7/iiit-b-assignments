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

# Preprocesses a given text document.
#   Args:
#     document (str): The input text document.
#   Returns:
#     str: The preprocessed text document.
#   This function performs the following preprocessing steps:
#     1. **Lowercasing:** Converts all characters to lowercase.
#     2. **Punctuation and Number Removal:** Removes punctuation marks and words containing numbers.
#     3. **Tokenization:** Splits the text into individual words.
#     4. **Stop Word Removal:** Removes common stop words (e.g., "the," "and," "of").
#     5. **Short Word Removal:** Removes words with one or fewer characters.
#     6. **Joining:** Joins the processed words back into a single string.
def preprocess(document):
  # 1. Change the document to lower case
  document = document.lower()

  # 2. Remove punctuation and words containing numbers
  document = re.sub("[^\sA-z]","",document)

  # 3. Tokenize the words
  words = word_tokenize(document)

  negative_words = {
        "no",
        "not",
        "none",
        "neither",
        "never",
        "nobody",
        "nothing",
        "nowhere",
        "doesn't",
        "isn't",
        "wasn't",
        "shouldn't",
        "won't",
        "can't",
        "couldn't",
        "don't",
        "haven't",
        "hasn't",
        "hadn't",
        "aren't",
        "weren't",
        "wouldn't",
        "daren't",
        "needn't",
        "didn't",
        "without",
        "against",
        "negative",
        "deny",
        "reject",
        "refuse",
        "decline",
        "unhappy",
        "sad",
        "miserable",
        "hopeless",
        "worthless",
        "useless",
        "futile",
        "disagree",
        "oppose",
        "contrary",
        "contradict",
        "disapprove",
        "dissatisfied",
        "objection",
        "unsatisfactory",
        "unpleasant",
        "regret",
        "resent",
        "lament",
        "mourn",
        "grieve",
        "bemoan",
        "despise",
        "loathe",
        "detract",
        "abhor",
        "dread",
        "fear",
        "worry",
        "anxiety",
        "sorrow",
        "gloom",
        "melancholy",
        "dismay",
        "disheartened",
        "despair",
        "dislike",
        "aversion",
        "antipathy",
        "hate",
        "disdain"
    }

  # 3. Tokenize the words
  words = word_tokenize(document)

  new_stop_set = stop_words = set(stopwords.words('english')) - set(negative_words)
  # 4. Remove the stop words
  words = [word for word in words if word not in new_stop_set]

  # 5. Remove words with 0 or 1 letter
  words = [w for w in words if len(w) > 1]

  # 6. Join the processed words back into a single string
  document = " ".join(words)
  return(document)



# Lemmatizes a given text using spaCy.
# Args:
# text (str): The input text to be lemmatized.
# Returns:
# str: The lemmatized text.
def lemmatize_text(text):
    sent = []
    
    # Process the text using spaCy's NLP pipeline
    doc = nlp(text) 
    for token in doc:
        # Append the lemma of each token to the sentence list
        sent.append(token.lemma_)
    
    # Join the lemmatized tokens into a single string and return the result
    return " ".join(sent)



# Gets the top N most frequent n-grams from a given corpus.
# Args:
#     corpus (list): A list of text documents.
#     n_gram_range (int): The range of n-grams to consider.
#     n (int, optional): The number of top n-grams to return. Defaults to None, which returns all n-grams.
# Returns:
#     list: A list of tuples, where each tuple contains an n-gram and its frequency.
def get_top_n_ngram( corpus, n_gram_range ,n=None):
    # Create a CountVectorizer to count the frequency of n-grams
    vec = CountVectorizer(ngram_range=(n_gram_range, n_gram_range), stop_words='english').fit(corpus)
    
    # Transform the corpus into a bag-of-words representation
    bag_of_words = vec.transform(corpus)

    # Sum the frequencies of each n-gram across all documents
    sum_words = bag_of_words.sum(axis=0)    # --1: Calculate the sum of frequencies for each n-gram 
    # print("--1",sum_words)

    # Get the first word and its index to break the loop
    for word, idx in vec.vocabulary_.items():
        break

    # Create a list of tuples, where each tuple contains an n-gram and its frequency
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    # Sort the list of tuples by frequency in descending order
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    # Return the top N n-grams
    return words_freq[:n]