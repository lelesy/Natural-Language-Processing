from nltk.corpus import brown
import nltk


def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features

tagged_sents = brown.tagged_sents(categories='news',tagset='universal')

featuresets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append( (pos_features(untagged_sent, i), tag) )

my_test = [
    [(u'Zhumaskaliyev','???'),(u'scores','VERB'),(u'a','DET'),(u'goal','NOUN')],
    [(u'He','NOUN'),(u'visited','VERB'),(u'Kaskelen','???'),(u'last','ADJ'),(u'month','NOUN')],
    [(u'They','NOUN'),(u'were','VERB'),(u'challing','???'),(u'in','ADP'),(u'cafe','NOUN')]
]
new_f = []
for tagged_sent in my_test:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        new_f.append( (pos_features(untagged_sent, i), tag) )


classifier = nltk.NaiveBayesClassifier.train(featuresets)

test_set = new_f

for item in test_set:
    print item
    print classifier.classify(item[0])
