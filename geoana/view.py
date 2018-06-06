import numpy as np
import matplotlib.pyplot as plt


def plotImage(
    x,
    y,
    z,
    v,
    view='amplitude',
    normal='z',
    index=None,
    fig=None,
    ax=None,
    colorbar=True,
    clim=None,
    pcolorOpts={},
    streamOpts={},
    showIt=False,
):

    # allowed inputs
    viewTypes = ['vec', 'amplitude', 'phase', 'real', 'imag', 'x', 'y', 'z']
    normalOpts = ['x', 'y', 'z']

    ################
    # Check inputs #
    ################

    # view
    assert type(v) is str, 'view must be a string'
    view = view.lower()

    [
        view.replace(vec_eqivalent, 'vec') for vec_eqivalent in
        ['vector', 'streamlines', 'streamplot']
    ]

    view = view.split()

    for v in view:
        assert v in viewTypes, "view must be in {}. You provided {}".format(
            viewTypes, view
        )

    # normal
    assert type(normal) is str, 'normal must be a string'
    normal = normal.lower()

    assert normal in normalOpts, (
        'normal must be in {}. You provided {}'.format(
            normalOpts, normal
        )
    )

    # figure out which axes to plot
    if len(x) == 1:




    ######################
    # Get vector to plot #
    ######################









