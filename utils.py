from pathlib import Path
from itertools import islice
import matplotlib.pyplot as plt
import PIL

def rm_dir(path):  # delete a directory
    if path.exists():
        for child in path.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                rm_dir(child)
        path.rmdir()

def plots(images, caption=None, output="image.jpg", n=5):
    import warnings
    warnings.filterwarnings("ignore", message="Glyph.* missing from current font")

    fig = plt.figure(figsize=(5*n, 20))
    for i, f in enumerate(islice(images, n)):
        ax = fig.add_subplot(1, n, i+1)
        ax.axis("off")
        ax.set_title(caption(i, f))
        plt.figimage(plt.imread(f))
    plt.savefig(output, bbox_inches='tight')
    return plt.imread(output) # fixme
#     return PIL.Image.open(output) # fixme

#     return fig
