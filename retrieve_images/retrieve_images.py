from gensim.models import Word2Vec
from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parents[1]))
import torch
import heapq as hq

from settings import Dir
import utils

def save_image_vectors(model, dataloader_test):
    utils.rm_dir(Dir.image_vectors)
    with torch.no_grad():
        model.eval()
        for img, _, img_name, _ in dataloader_test:
            vectors = model(img.to("cuda"))
            for i, v in enumerate(vectors):
                path = Dir.image_vectors / Path(img_name[i]).relative_to(Dir.images)
                path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(v, str(path))

                
def plot_closest(query, n_results = 5):
    word2vec = Word2Vec.load(str(Dir.model_embed_words))
    query_vector = torch.from_numpy(word2vec.wv.get_vector(query)).to("cuda")
    closest = []

    for file_img in Dir.image_vectors.rglob("*.jpg"):
        v = torch.load(file_img)
        d = ((v - query_vector)**2).sum(axis=0).item()
        if len(closest) < n_results:
            hq.heappush(closest, (-d, file_img))
        elif -closest[0][0] > d:
            hq.heappushpop(closest, (-d, file_img))

    dist, images = zip(*sorted(closest, key=lambda x: -x[0]))
    return utils.plots([Dir.images / img.relative_to(Dir.image_vectors) for img in images], lambda i, _: -dist[i])
