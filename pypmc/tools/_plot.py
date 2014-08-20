def plot_mixture(mixture, i=0, j=1, center_style=dict(s=0.15),
                 cmap='spectral', cutoff=0.0, ellipse_style=dict(alpha=0.3),
                 solid_edge=True):
    '''Plot the (Gaussian) components of the ``mixture`` density as
    one-sigma ellipses in the ``(i,j)`` plane.

    :param center_style:
        If a non-empty ``dict``, plot mean value with the style passed to ``scatter``.

    :param cmap:

        The color map to which components are mapped in order to
        choose their facecolor. The facecolor only depends on the
        index and total number of components. It is unaffected by the
        ``cutoff``.

    :param cutoff:
        Ignore components whose weight is below the ``cut off``.

    :param ellipse_style:
        Passed on to define the properties of the ``Ellipse``.

    :param solid_edge:
        Draw the edge of the ellipse as solid opaque line.

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
    colors = [cmap(k) for k in np.linspace(0, 0.9, len(mixture.components))]

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
