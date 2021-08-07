from pathlib import Path
import multiprocessing

# @dataclass
# class Train:
    
class Params:
    dim_embedding = 40  # dimension of the word embedding
    samples = 50_000  # limit number of samples per city to speed up
    workers = multiprocessing.cpu_count()
    
class Dir:
    root = Path(__file__).parent
    
    data_name = "newyork"
    data = root / "data_ny"
#     data = Path("/media/qfortier/c796cdda-d6ec-4dae-bcd9-50ef6b2e81b2/data_instagram")
#     captions = root / "data_instagram" / "captions"
    captions = data / "captions"
    images = data / "images"
    
    embed_words = root / "embed_words"
    caption_vectors = embed_words / "vectors"
    model_embed_words = embed_words / "model" / f"word2vec_{Params.dim_embedding}_{Params.samples}"
    
    model_cnn = root / "embed_images" / "models" / "resnet18_ny.pt"
    image_vectors = root / "embed_images" / "vectors"