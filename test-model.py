from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from joblib import dump

# preprocessing data
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    words = text.split()
    processed_words = []
    for word in words:
        if word not in stop_words:
            stemmed = stemmer.stem(word)
            lemmatized = lemmatizer.lemmatize(stemmed)
            processed_words.append(lemmatized)
    return ' '.join(processed_words)

model = load('sentiment_model.joblib')
vect = load('count_vectorizer.joblib')

reviews = [
    "Despite low expectations, this film surprised me in a positive way. The plot seemed predictable, but the plot twists were well thought out and the execution flawless.",
    "I was initially skeptical about the choice of cast, but I must admit that each actor brought unexpected depth to their characters. A film that deserves more attention.",
    "This film was nothing like I expected, and that's a good thing. It exceeded my expectations in every aspect, making it a unique experience.",
    "On the surface, it looks like the usual unpretentious film, but it hides an emotional richness and narrative that completely captures you. Absolutely a must-see.",
    "The director plays with genre clichés in a clever way, turning a potentially banal plot into a work that invites reflection. Kudos to the director for daring.",
    "The film attempts to be profound, but ends up drowning in its own complexity. Too many themes treated superficially, leaving the viewer more confused than satisfied.",
    "What could have been a masterpiece is lost in a fragmented narrative and underdeveloped characters. A disappointment considering the potential.",
    "Despite some good performances, the film is weighed down by a weak script that fails to keep the viewer glued to the screen.",
    "A film that tries to be different but ends up falling into the same old traps. It lacks originality and real emotional impact.",
    "A film that you forget as soon as you leave the cinema. It leaves the viewer with nothing but the question of what was missing to make it memorable.",
]

for review in reviews:
    review_pre = preprocess_text(review)
    review_vect = vect.transform([review_pre])
    prediction = model.predict(review_vect)
    print(f"Review: \"{review}...\"")
    print(f"Pred: {prediction}\n")
