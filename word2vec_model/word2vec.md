# Embed captions with word2vec

## Load and preprocess captions

```python
%load_ext autoreload
%autoreload 2

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import re
import itertools

dir_root = Path().resolve().parent
import sys; sys.path.append(str(dir_root))
from settings import Dir, Params
```

```python
def remove_useless(words):
    return [w for w in words if w not in ['http','https','photo','picture','image','insta','instagram','instagood','post']]

def preprocess(w): # remove stop words, hashtags, non-ASCII and lower characters from w
    return remove_stopwords(re.sub("[^a-zA-Z ]", '', w.replace('#', ' ')).lower())

class Captions:  # iterator for captions
    def __iter__(self):
        for f in Dir.captions.rglob("*.txt"):
            yield remove_useless(preprocess(f.read_text()).split())
```

```python
for c in itertools.islice(Captions(), 5):
    print(c)  # show some captions
```

## Train a word2vec model

```python
model = Word2Vec(sentences=Captions(),
                 vector_size=Params.dim_embedding,
                 min_count=10)
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
model.wv.most_similar(positive=['food'] , negative=['healthy'])
```

```python
model.wv.similar_by_vector(model.wv.get_vector('actress') + model.wv.get_vector('woman'), topn=15)
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
    for w in caption.split():
        if len(w) > 0 and w[0] == '#':
            if w[1:] in wv.key_to_index:
                return wv.get_vector(w[1:])
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
rm_tree(Dir.caption_vectors)
n_vectors = 100000 # to speed up training

for dir_city in Dir.captions.iterdir():
    for n, file_caption in itertools.islice(enumerate(dir_city.iterdir()), n_vectors): 
        v = representation(model.wv, file_caption.read_text())
        if v is not None:
            mode = "train"
            if n > .7*n_vectors:
                mode = "validate"
            if n >= .8*n_vectors:
                mode = "test"
            file = Dir.caption_vectors / f"{mode}/{dir_city.name}/{file_caption.name}"
            Path(file.parent).mkdir(parents=True, exist_ok=True)
            np.savetxt(str(file), v)
```

```python

```
