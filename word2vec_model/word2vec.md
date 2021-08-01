# Embed captions with word2vec

## Load and preprocess captions

```python
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import re
import itertools

dir_root = Path().resolve().parent
dir_captions = dir_root / "data" / "captions"
dir_vectors = Path().resolve() / "vectors"
```

```python
def remove_useless(words):
    return [w for w in words if w not in ['http','https','photo','picture','image','insta','instagram','post']]

def preprocess(w): # remove stop words, hashtags, non-ASCII and lower characters from w
    return remove_stopwords(re.sub("[^a-zA-Z ]", '', w.replace('#', ' ')).lower())

class Captions:  # iterator for captions
    def __iter__(self):
        for f in Path(dir_captions).rglob("*.txt"):
            yield preprocess(f.read_text()).split() # todo remove insta...
```

```python

for c in itertools.islice(Captions(dir_captions), 5):
    print(c)  # show some captions
```

## Train a word2vec model

```python
model = Word2Vec(sentences=Captions(),
                 vector_size=40,
                 min_count=5,
                 epochs=25,
                 window=8)
model.save("model_captions")
```

```python
model = Word2Vec.load("model_captions")
```

```python
print(f"Number of captions: {model.corpus_count}")
print(f"Number of words: {len(model.wv)}")
```

```python
pd.DataFrame({"caption": model.wv.index_to_key, 
              "count": map(lambda w: model.wv.get_vecattr(w, "count"), model.wv.index_to_key)
             }) \
  .set_index("caption") \
  .iloc[:20].T  # 20 most frequent words
```

## Explore results

```python
model.wv.most_similar("fashion")
```

```python
model.wv.most_similar(positive=['food'] , negative=['natural'])
```

```python
model.wv.similar_by_vector(model.wv.get_vector('actor') + model.wv.get_vector('woman'), topn=15)
```

```python
from sklearn.manifold import TSNE
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
    
def reduce_2d(model):
    vectors = TSNE(n_components=2).fit_transform(np.asarray(model.wv.vectors[10:50]))
    return [v[0] for v in vectors], [v[1] for v in vectors], np.asarray(model.wv.index_to_key[10:50])

x_vals, y_vals, labels = reduce_2d(model)
init_notebook_mode(connected=True)
iplot([go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)])
```

## Save vectors

```python
def representation(wv, caption):  # return the vector representation of caption
    words = [wv.get_vector(w) for w in preprocess(caption) if w in wv.key_to_index]
    if len(words) > 0:
        v = np.sum(words, axis=0) / len(words)
        return v
```

```python
def rm_tree(path):
    if path.exists():
        for child in path.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        path.rmdir()
```

```python
rm_tree(dir_vectors)
n_vectors = 10000 # to speed up training

for dir_city in dir_captions.iterdir():
    for n, file_caption in itertools.islice(enumerate(dir_city.iterdir()), n_vectors): 
        v = representation(model.wv, file_caption.read_text())
        if v is not None:
            mode = "train"
            if n > .7*n_vectors:
                mode = "validate"
            if n >= .8*n_vectors:
                mode = "test"
            file = dir_vectors / f"{mode}/{dir_city.name}/{file_caption.name}"
            Path(file.parent).mkdir(parents=True, exist_ok=True)
            np.savetxt(str(file), v)
```

```python

```
