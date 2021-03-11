_max_color = 0.9

def plot_mixture(mixture, i=0, j=1, center_style=dict(s=0.15),
                 cmap='nipy_spectral', cutoff=0.0, ellipse_style=dict(alpha=0.3),
                 solid_edge=True, visualize_weights=False):
    '''Plot the (Gaussian) components of the ``mixture`` density as
    one-sigma ellipses in the ``(i,j)`` plane.

    :param center_style:
        If a non-empty ``dict``, plot mean value with the style passed to ``scatter``.

    :param cmap:

        The color map to which components are mapped in order to
        choose their face color. It is unaffected by the
        ``cutoff``. The meaning depends on ``visualize_weights``.

    :param cutoff:
        Ignore components whose weight is below the ``cut off``.

    :param ellipse_style:
        Passed on to define the properties of the ``Ellipse``.

    :param solid_edge:
        Draw the edge of the ellipse as solid opaque line.

    :param visualize_weights:
        Colorize the components according to their weights if ``True``.
        One can do `plt.colorbar()` after this function and the bar allows to read off the weights.
        If ``False``, coloring is based on the component index and the total number of components.
        This option makes it easier to track components by assigning them the same color in subsequent calls to this function.

    '''
    # imports inside the function because then "ImportError" is raised on
    # systems without 'matplotlib' only when 'plot_mixture' is called
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.cm import get_cmap

    assert i >= 0 and j >= 0, 'Invalid submatrix specification (%d, %d)' % (i, j)
    assert i != j, 'Identical dimension given: i=j=%d' % i
    assert mixture.dim >= 2, '1D plot not supported'

    cmap = get_cmap(name=cmap)

    if visualize_weights:
        # colors according to weight
        renormalized_component_weights  = np.array(mixture.weights)
        colors = [cmap(k) for k in renormalized_component_weights]
    else:
        # colors according to index
        colors = [cmap(k) for k in np.linspace(0, _max_color, len(mixture.components))]

    mask = mixture.weights >= cutoff

    # plot component means
    means = np.array([c.mu for c in mixture.components])
    x_values = means.T[i]
    y_values = means.T[j]

    for k, w in enumerate(mixture.weights):
        # skip components by hand to retain consistent coloring
        if w < cutoff:
            continue

        cov = mixture.components[k].sigma
        submatrix = np.array([[cov[i,i], cov[i,j]], \
                              [cov[j,i], cov[j,j]]])

        # for idea, check
        # 'Combining error ellipses' by John E. Davis
        correlation = np.array([[1.0, cov[i,j] / np.sqrt(cov[i,i] * cov[j,j])], [0.0, 1.0]])
        correlation[1,0] = correlation[0,1]

        assert abs(correlation[0,1]) <= 1, 'Invalid component %d with correlation %g' % (k, correlation[0, 1])

        ew, ev = np.linalg.eigh(submatrix)
        assert ew.min() > 0, 'Nonpositive eigenvalue in component %d: %s' % (k, ew)

        # rotation angle of major axis with x-axis
        if submatrix[0,0] == submatrix[1,1]:
            theta = np.sign(submatrix[0,1]) * np.pi / 4.
        else:
            theta = 0.5 * np.arctan( 2 * submatrix[0,1] / (submatrix[1,1] - submatrix[0,0]))

        # put larger eigen value on y'-axis
        height = np.sqrt(ew.max())
        width = np.sqrt(ew.min())

        # but change orientation of coordinates if the other is larger
        if submatrix[0,0] > submatrix[1,1]:
            height = np.sqrt(ew.min())
            width = np.sqrt(ew.max())

        # change sign to rotate in right direction
        angle = -theta * 180 / np.pi

        # copy keywords but override some
        ellipse_style_clone = dict(ellipse_style)

        # overwrite facecolor
        ellipse_style_clone['facecolor'] = colors[k]

        ax = plt.gca()

        # need full width/height
        e = Ellipse(xy=(x_values[k], y_values[k]),
                                   width=2*width, height=2*height, angle=angle,
                                   **ellipse_style_clone)
        ax.add_patch(e)

        if solid_edge:
            ellipse_style_clone['facecolor'] = 'none'
            ellipse_style_clone['edgecolor'] = colors[k]
            ellipse_style_clone['alpha'] = 1
            ax.add_patch(Ellipse(xy=(x_values[k], y_values[k]),
                                       width=2*width, height=2*height, angle=angle,
                                       **ellipse_style_clone))

    if center_style:
        plt.scatter(x_values[mask], y_values[mask], **center_style)

    if visualize_weights:
        # to enable plt.colorbar()
        mappable = plt.gci()
        mappable.set_array(mixture.weights)
        mappable.set_cmap(cmap)

def plot_responsibility(data, responsibility,
                        cmap='nipy_spectral'):
    '''Classify the 2D ``data`` according to the ``responsibility`` and
    make a scatter plot of each data point with the color of the
    component it is most likely from. The ``responsibility`` is
    normalized internally such that each row sums to unity.

    :param data:

        matrix-like; one row = one 2D sample

    :param responsibility:

        matrix-like; one row = probabilities that sample n is from
        1st, 2nd, ... component. The number of rows has to agree with ``data``

    :param cmap:

        colormap; defines how component indices are mapped to the
        color of the data points

    '''
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.cm import get_cmap

    data = np.asarray(data)
    responsibility = np.asarray(responsibility)

    assert data.ndim == 2
    assert responsibility.ndim == 2

    D = data.shape[1]
    N = data.shape[0]
    K = responsibility.shape[1]

    assert D == 2
    assert N == responsibility.shape[0]

    # normalize responsibility so each row sums to one
    inv_row_sum = 1.0 / np.einsum('nk->n', responsibility)
    responsibility = np.einsum('n,nk->nk', inv_row_sum, responsibility)

    # index of the most likely component for each sample
    indicators = np.argmax(responsibility, axis=1)

    # same color range as in plot_mixture
    if K > 1:
        point_colors = indicators / (K - 1) * _max_color
    else:
        point_colors = np.zeros(N)
    plt.scatter(data.T[0], data.T[1], c=point_colors, cmap=cmap)
