

```python
import re
from pathlib import Path

dir_root = Path().resolve().parent
dir_captions = dir_root / "data" / "captions"
```

```python
from gensim.parsing.preprocessing import remove_stopwords

def preprocess(w):  # remove non-ASCII and lower characters from w
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

```python
from gensim.models import Word2Vec

captions = Captions(dir_captions)
model = Word2Vec(sentences=captions,
                 vector_size=20,
                 window=4, 
                 min_count=10, 
                 workers=12)
model.save("model_captions")
```

```python
model.wv.get_vecattr("sea", "count")
```

```python
model.cum_table
```

```python
import pandas as pd

pd.DataFrame({"caption": model.wv.index_to_key, 
              "count": map(lambda w: model.wv.get_vecattr(w, "count"), model.wv.index_to_key)
             }) \
  .set_index("caption") \
  .iloc[:20].T  # 20 most frequent words
```

```python
model.wv.most_similar("fashion")
```

```python
model.wv.most_similar(positive=['sea'] , negative=['drink'])
```
