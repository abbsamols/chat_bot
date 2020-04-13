# Chat bot created by Samuel Olsson

from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings
from os import system, name
from difflib import SequenceMatcher

warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('wordnet')

sent_tokens = []

remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text): # Gör en lista med artikelns ord med bara små bokstäver.
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

GREETING_INPUTS = ["hej", "hallå", "tjena", "goddag", "godmorgon", "godkväll"] # Hälsningsfraser från användaren.
GREETING_RESPONSES = ["hej", "hallå", "tjena"] # Hälsningsfraser som kan slumpas fram av AI:n.

def greeting(sentence): # Slumpar hälsningsfras.
    for word in sentence.split(): # Om användarens inmatning har minst ett ord som definieras som hälsningsfras kommer AI:n att hälsa tillbaka.
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    bot_response = ''
    questions = ["berätta om björnar?", "berätta om katter?", "berätta om hundar?"] # Frågor som AI:n utgår från för att avgöra vad användaren frågar efter.
    articles = ['https://sv.wikipedia.org/wiki/Bj%C3%B6rnar', 'https://sv.wikipedia.org/wiki/Katt', 'https://sv.wikipedia.org/wiki/Hund'] # Länkar som AI:n hämtar information från för att svara på användarens frågor.
    results = []
    index = 0
    for question in questions:
        Question_similarity = SequenceMatcher(a=questions[index], b=user_response).ratio() # Avgör hur lik användarens fråga är de frågor som AI:n utgår ifrån.
        results.append(Question_similarity) # Resultaten för hur lik frågan är läggs till i en lista.
        index += 1
    results_sorted = sorted(results, key=None, reverse=True)
    index = 0
    for result in results: # Går igenom alla resultat och kollar om det är det bästa resultatet.
        if result == results_sorted[0]:
            article = Article(articles[index]) # Bestämmer vilken länk där information som bäst svarar på användarens fråga finns.
            article.download()
            article.parse()
            article.nlp()
            text = article.text

            sent_tokens = nltk.sent_tokenize(text) #Konverterar artikelns text till en lista med meningar.
            remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)
            break
        else:
            index += 1

    sent_tokens.append(user_response)

    TfidVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')

    tfidf = TfidVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf) # Bestämmer hur lik användarens input är de olika meningarna i artikeln.
    idx = vals.argsort()[0][-2] # Bestämmer meningen som är mest lik användarens input.
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]

    if score < 0.1:
        bot_response =  bot_response + "Jag förstår tyvärr inte."
    else:
        bot_response = bot_response + sent_tokens[idx] # Om "score" är större än 0.1 svarar boten på användarens fråga."
    sent_tokens.remove(user_response)
    return bot_response

on = True
print("Bot: Hej! Jag är en bot som kan svara på frågor. För att stänga av mig, skriv hejdå!")
while on == True:
    user_response = input().lower()
    if user_response != "hejdå":
        if user_response == "tack" or user_response == "tackar":
            print("Bot: Varsågod!")
        else:
            if greeting(user_response) != None:
                print("Bot: " + greeting(user_response))
            else:
                print("Bot: " + response(user_response)) # Svarar på användarens fråga.
    else:
        on = False
        print("Bot: Hejdå!")