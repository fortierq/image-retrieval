# To add a new cell, type ''
# To add a new markdown cell, type ''

# # Load and preprocess data 


stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])



from os import mkdir
from pathlib import Path
import re

def rm_stop(words):  # remove stop words from list of words
    return [w for w in words.split() if w not in stop_words]

def preprocess(w):  # remove non-ASCII and lower characters from w
    return rm_stop(re.sub("[^a-zA-Z ]", '', w).lower())

class InstaCities:
    def __iter__(self):
        for f in (Path(__file__).parents[1] / "InstaCities1M/captions_resized_1M/cities_instagram").rglob("*.txt"):
            yield preprocess(f.read_text())


# # Word2Vec model

import gensim.models

sentences = InstaCities()

model = gensim.models.Word2Vec(sentences=sentences, vector_size=20, window=4, min_count=10, workers=12)
model.save("model_insta_cities")



wv = model.wv
wv.most_similar('city', topn=10)


# # Comparison with Google News model


import gensim.downloader as api
wv_news = api.load('word2vec-google-news-300')



from pprint import pprint

pprint(wv_news.most_similar(positive=["car", "sports", "hairstyle", "food", "toronto"], topn=5))
pprint(model.wv.most_similar(positive=["car", "sports", "hairstyle", "food", "toronto"], topn=5))



import numpy as np

top = 7

def similar(wv, *words):
    v = np.sum([wv.get_vector(w) for w in words], axis=0)
    return [w[0] for w in wv.similar_by_vector(v, topn=top)]

for w1, w2 in [("toronto", "wild"), ("happy", "family"), ("food", "healthy"), ("food", "sweet")]:
    print(f"Most similar words to {w1} + {w2}:")
    print(f"InstaCities: {similar(wv, w1, w2)}")
    print(f"Google News: {similar(wv_news, w1, w2)}\n")

print(f"Most similar words to food - healthy:")
print(f"InstaCities: {[w[0] for w in wv.similar_by_vector(wv.get_vector('food') - wv.get_vector('healthy'), topn=top)]}")
print(f"Google News: {[w[0] for w in wv_news.similar_by_vector(wv_news.get_vector('food') - wv_news.get_vector('healthy'), topn=top)]}")
print([w[0] for w in model.wv.most_similar(positive=['food'] , negative=['healthy'], topn=top)])



print(wv.doesnt_match(["man", "woman", "kid", "dog"]))
print(wv_news.doesnt_match(["man", "woman", "kid", "dog"]))



print(wv.similarity('woman', 'man'))
print(wv_news.similarity('woman', 'man'))

def repr(caption):
    words = preprocess(caption)
    words = [wv.get_vector(w) for w in words if w in wv.key_to_index]
    if len(words) > 0:
        v = np.sum(words, axis=0)
        v -= min(v)
        if max(v) != 0:
            v /= max(v)
        return v


for dir_city in (Path(__file__).parents[1] / "InstaCities1M/captions_resized_1M/cities_instagram").iterdir():
    n = 0
    for file_img in dir_city.iterdir():
        v = repr(file_img.read_text())
        if v is not None:
            mode = "train"
            if 85_000 > n > 80_000:
                mode = "validate"
            elif n >= 85_000:
                mode = "test"
            file = Path(__file__).parent / f"caption/{mode}/{dir_city.name}/{file_img.name}"
            Path(file.parent).mkdir(parents=True, exist_ok=True)
            np.savetxt(str(file), v)
            n += 1
