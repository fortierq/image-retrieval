from pathlib import Path

class Params:
    dim_embedding = 200
    
class Dir:
    root = Path(__file__).parent
    
    data = root / "data"
    captions = data / "captions"
    images = data / "img_resized"
    
    word_embedding = root / "word2vec_model"
    caption_vectors = word_embedding / "vectors"
    
    image_vectors = root / "cnn_text_supervision" / "vectors"