#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll


def matrix_2d_effect(transfo_matrix, vectorsCol=["#FFD800", "#00CD79"]):
    """
    Modify the unit circle and basis vector by applying a matrix.
    Visualize the effect of the matrix in 2D.
    Parameters
    ----------
    transfo_matrix : array-like
        2D matrix to apply to the unit circle.
    vectorsCol : HEX color code
        Color of the basis vectors
    Returns:
    fig : instance of matplotlib.figure.Figure
        The figure containing modified unit circle and basis vectors.
    """
    # Unit circle
    x = np.linspace(-0.9998, 0.9998, 100000)
    y = np.sqrt(1 - (x ** 2))

    # Modified unit circle (separate negative and positive parts)
    x1 = transfo_matrix[0, 0] * x + transfo_matrix[0, 1] * y
    y1 = transfo_matrix[1, 0] * x + transfo_matrix[1, 1] * y
    x1_neg = transfo_matrix[0, 0] * x - transfo_matrix[0, 1] * y
    y1_neg = transfo_matrix[1, 0] * x - transfo_matrix[1, 1] * y

    # Vectors
    vecs = np.array([transfo_matrix[:, 0], transfo_matrix[:, 1]])

    ax = plt.axes()
    ax.axvline(x=0, color="#d6d6d6", zorder=0)
    ax.axhline(y=0, color="#d6d6d6", zorder=0)

    plotVectors(vecs, ax, cols=vectorsCol, alpha=1)

    ax.plot(x1, y1, "#F57F53", linewidth=4, alpha=1)
    ax.plot(x1_neg, y1_neg, "#F57F53", linewidth=4, alpha=1)
    ax.axis("equal")
    return ax


def matrix_3_by_2_effect(transfo_matrix, vectorsCol=["#FFD800", "#00CD79"]):
    """
    Modify the unit circle by applying a matrix.
    Visualize the effect of the matrix in 3D.
    Parameters
    ----------
    transfo_matrix : array-like
        3 by 2 matrix to apply to the unit circle.
    Returns:
    fig : instance of matplotlib.figure.Figure
        The figure containing modified unit circle.
    """
    theta = np.arange(0, 2 * np.pi, 0.1)
    r = 1
    x1 = r * np.cos(theta)

    x2 = r * np.sin(theta)

    new_x = transfo_matrix @ np.array([x1, x2])

    ax = plt.axes(projection="3d")

    for i in range(new_x.shape[1]):
        ax.scatter(new_x[0, i], new_x[1, i], new_x[2, i], c="#2EBCE7", alpha=0.3)

    # Plot basis vectors
    for i in range(transfo_matrix.shape[1]):
        plt.quiver(
            0,
            0,
            0,
            transfo_matrix[0, i],
            transfo_matrix[1, i],
            transfo_matrix[2, i],
            color=vectorsCol[i],
            arrow_length_ratio=0.2,
            alpha=0.5,
        )
    return ax


def plotVectors(vecs, ax, cols, alpha=1):
    """
    Plot set of vectors.
    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors
    Returns:
    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    ax.quiver(
        np.zeros(vecs.shape[0]),
        np.zeros(vecs.shape[0]),
        vecs[:, 0],
        vecs[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=cols,
        width=0.018,
        alpha=alpha,
    )


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

