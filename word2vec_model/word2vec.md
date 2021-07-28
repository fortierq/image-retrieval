# Embed captions with word2vec

## Load and preprocess captions

```python
import re
from pathlib import Path

dir_root = Path().resolve().parent
dir_captions = dir_root / "data" / "captions"
```

```python
from gensim.parsing.preprocessing import remove_stopwords

def preprocess(w):  # remove stop words, non-ASCII and lower characters from w
    return remove_stopwords(re.sub("[^a-zA-Z ]", '', w).lower())
```

```python
class Captions:  # iterator for captions
    def __init__(self, path_dir):
        self.path_dir = path_dir
        
    def __iter__(self):
        for f in Path(self.path_dir).rglob("*.txt"):
            yield preprocess(f.read_text()).split()
```

```python
import itertools

for c in itertools.islice(Captions(dir_captions), 5):
    print(c)  # show some captions
```

## Train a word2vec model

```python
from gensim.models import Word2Vec

captions = Captions(dir_captions)
model = Word2Vec(sentences=captions,
                 vector_size=50,
                 window=4, 
                 min_count=10, 
                 workers=12)
model.save("model_captions")
```

```python
model.corpus_count  # number of captions
```

```python
import pandas as pd

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
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def reduce_2d(model):
    vectors = TSNE(n_components=2).fit_transform(np.asarray(model.wv.vectors))
    return [v[0] for v in vectors], [v[1] for v in vectors], np.asarray(model.wv.index_to_key)

x_vals, y_vals, labels = reduce_2d(model)

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)

```

## Save vectors

```python
def repr(caption):  # return a vector representation of a caption
    words = [wv.get_vector(w) for w in preprocess(caption) if w in wv.key_to_index]
    if len(words) > 0:
        v = np.sum(words, axis=0)  # sum of vector representations of words
        v -= min(v)
        if max(v) != 0:
            v /= max(v)
        return v
```

```python
for dir_city in dir_root / "InstaCities1M/captions_resized_1M/cities_instagram").iterdir():
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
```
