from pathlib import Path
import multiprocessing

class Params:
    dim_embedding = 100  # dimension of the word embedding
    samples = 50_000  # limit number of samples per city to speed up
    workers = multiprocessing.cpu_count()
    
class Dir:
    root = Path(__file__).parent
    
    data = Path("/media/qfortier/c796cdda-d6ec-4dae-bcd9-50ef6b2e81b2/data_instagram")
    captions = root / "data_instagram" / "captions"
    images = data / "images"
    
    word_embedding = root / "word_embedding"
    caption_vectors = word_embedding / "vectors"
    model_embedding = word_embedding / "model" / f"word2vec_{Params.dim_embedding}_{Params.samples}"
    
    image_vectors = root / "cnn_training" / "vectors"