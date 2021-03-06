## the topic of manifold learning, also called nonlinear dimensionality reduction
# 高维数据映射到低维,数据在空间的分布主要有两个特性,一个是相似性，用类内距离衡量;一个是差异性,用类间距离衡量


import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
# %matplotlib inline    # jupyter中使用

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    return f, ax, sc, txts


def _joint_probabilities_constant_sigma(D, sigma):
    P = np.exp(-D**2/2 * sigma**2)
    P /= np.sum(P, axis=1)
    return P


if __name__ == '__main__':
    digits = load_digits()
    digits.data.shape

    print(digits['DESCR'])

    nrows, ncols = 2, 5
    plt.figure(figsize=(6,3))
    plt.gray()
    for i in range(ncols * nrows):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.matshow(digits.images[i,...])
        plt.xticks([]); plt.yticks([])
        plt.title(digits.target[i])
    plt.savefig('images/digits-generated.png', dpi=150)

    # We first reorder the data points according to the handwritten numbers.
    X = np.vstack([digits.data[digits.target==i]
                   for i in range(10)])
    y = np.hstack([digits.target[digits.target==i]
                   for i in range(10)])
    digits_proj = TSNE(random_state=RS).fit_transform(X)

    scatter(digits_proj, y)
    plt.savefig('images/digits_tsne-generated.png', dpi=120)


    # Pairwise distances between all data points.
    D = pairwise_distances(X, squared=True)
    # Similarity with constant sigma.
    P_constant = _joint_probabilities_constant_sigma(D, .002)
    # Similarity with variable sigma.
    P_binary = _joint_probabilities(D, 30., False)
    # The output of this function needs to be reshaped to a square matrix.
    P_binary_s = squareform(P_binary)

    plt.figure(figsize=(12, 4))
    pal = sns.light_palette("blue", as_cmap=True)

    plt.subplot(131)
    plt.imshow(D[::10, ::10], interpolation='none', cmap=pal)
    plt.axis('off')
    plt.title("Distance matrix", fontdict={'fontsize': 16})

    plt.subplot(132)
    plt.imshow(P_constant[::10, ::10], interpolation='none', cmap=pal)
    plt.axis('off')
    plt.title("$p_{j|i}$ (constant $\sigma$)", fontdict={'fontsize': 16})

    plt.subplot(133)
    plt.imshow(P_binary_s[::10, ::10], interpolation='none', cmap=pal)
    plt.axis('off')
    plt.title("$p_{j|i}$ (variable $\sigma$)", fontdict={'fontsize': 16})
    plt.savefig('images/similarity-generated.png', dpi=120)


    ## t-Distribution
    npoints = 1000
    plt.figure(figsize=(15, 4))
    for i, D in enumerate((2, 5, 10)):
        # Normally distributed points.
        u = np.random.randn(npoints, D)
        # Now on the sphere.
        u /= norm(u, axis=1)[:, None]
        # Uniform radius.
        r = np.random.rand(npoints, 1)
        # Uniformly within the ball.
        points = u * r**(1./D)
        # Plot.
        ax = plt.subplot(1, 3, i+1)
        ax.set_xlabel('Ball radius')
        if i == 0:
            ax.set_ylabel('Distance from origin')
        ax.hist(norm(points, axis=1),
                bins=np.linspace(0., 1., 50))
        ax.set_title('D=%d' % D, loc='left')
    plt.savefig('images/spheres-generated.png', dpi=100, bbox_inches='tight')