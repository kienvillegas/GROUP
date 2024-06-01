import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag 

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

def print_separator():
    print("------------------------------------------------")

def tokenization(sentence):
    tokenize_sentence = word_tokenize(sentence)
    sentence_no_punc = sentence.translate(str.maketrans("", "", string.punctuation))
    tokenize_no_punc = word_tokenize(sentence_no_punc)
    print("Tokenization: ", tokenize_sentence)

    print_separator()
    stop_word_removal(tokenize_sentence)
    print_separator()
    stemming(tokenize_no_punc)
    print_separator()
    lemmatization(tokenize_no_punc)
    print_separator()
    part_of_speech_tagging(tokenize_sentence, sentence)
    print_separator()

def stop_word_removal(tokenize_sentence):
    stop_words = set(stopwords.words("english"))
    filtered_sentence = [word for word in tokenize_sentence if not word.lower() in stop_words]
    print("Stop word removal: ", filtered_sentence)

def stemming(tokenize_no_punc):
    ps = PorterStemmer()
    print("{0:20}{1:20}".format("--Word--","--Stem--"))
    for word in tokenize_no_punc:
        print("{0:20}{1:20}".format(word, ps.stem(word)))

def lemmatization(tokenize_no_punc):
    wnl = WordNetLemmatizer()
    print("{0:20}{1:20}".format("--Word--","--Lemma--"))
    for word in tokenize_no_punc:
        print("{0:20}{1:20}".format(word, wnl.lemmatize(word, pos="v")))

def part_of_speech_tagging(tokenize_sentence, sentence):
    pos_tags = pos_tag(tokenize_sentence)
    print("Original Text: ", sentence)
    print("PoS Tagging Result: ")
    for words, tag in pos_tags:
        print(f"{words}: {tag}")

print_separator()
sentence = input("Enter a sentence: ")
print_separator()
tokenization(sentence)