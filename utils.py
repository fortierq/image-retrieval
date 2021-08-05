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

def plots(images, caption=None, n=5):
    import warnings
    warnings.filterwarnings("ignore", message="Glyph.* missing from current font")

    fig = plt.figure(figsize=(5*n, 20))
    plt.tight_layout()
    for i, f in enumerate(islice(images, n)):
        ax = fig.add_subplot(1, n, i+1, aspect = 'equal')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.axis("off")
        ax.set_title(caption(i, f))
        plt.imshow(plt.imread(f))
    plt.close()
    return fig
