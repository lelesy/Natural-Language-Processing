import nltk.corpus,re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
from sklearn.cluster import KMeans
num_clusters = 4
km = KMeans(n_clusters=num_clusters)

transformer = TfidfTransformer(smooth_idf=True)
stop = set(nltk.corpus.stopwords.words('english'))

nameOfFile = ["001.txt","002.txt","003.txt","004.txt","005.txt","006.txt","007.txt","008.txt","009.txt","010.txt"]
mainContent = []

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

for i in range(10):
    filename = 'bbc/'+nameOfFile[i]
    tempArr = []
    with open(filename) as f:
        tempContent = f.read()
    mainContent.append(tempContent)

X_train_counts = CountVectorizer(stop_words='english',analyzer='word',tokenizer=tokenize_and_stem).fit_transform(mainContent)
X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)

km.fit(X_train_tfidf)
clusters = km.labels_.tolist()
print clusters




#
# (array([1,1,1,1]),1),
#     (array([1,1,0,0]),0),
#     (array([1,0,1,1]),0),
#     (array([0,0,0,1]),1),
#     (array([0,1,1,1]),1),
#     (array([0,0,0,0]),0),
