"""
=================================================
Basic Utilities (:mod:`geoana.utils`)
=================================================
.. currentmodule:: geoana.utils

The ``geoana.utils`` module provides a set of rudimentary functions that are
commonly used within the code base.

.. autosummary::
  :toctree: generated/

  mkvc
  ndgrid

"""
import numpy as np

def mkvc(x, n_dims=1):
    """Creates a vector with specified dimensionality.

    This function converts a :class:`numpy.ndarray` to a vector. In general,
    the output vector has a dimension of 1. However, the dimensionality
    can be specified if the user intends to carry out a dot product with
    a higher order array.

    Parameters
    ----------
    x : array_like
        An array that will be reorganized and output as a vector. The input array
        will be flattened on input in Fortran order.
    n_dims : int
        The dimension of the output vector. :data:`numpy.newaxis` are appened to the
        output array until it has this many axes.

    Returns
    -------
    numpy.ndarray
        The output vector, with at least ``n_dims`` axes.

    Examples
    --------
    Here, we reorganize a simple 2D array as a vector and demonstrate the
    impact of the *n_dim* argument.

    >>> from geoana.utils import mkvc
    >>> import numpy as np

    >>> a = np.random.rand(3, 2)
    >>> a
    array([[0.33534155, 0.25334363],
           [0.07147884, 0.81080958],
           [0.85892774, 0.74357806]])

    >>> v = mkvc(a)
    >>> v
    array([0.33534155, 0.07147884, 0.85892774, 0.25334363, 0.81080958,
           0.74357806])

    In Higher dimensions:

    >>> for ii in range(1, 4):
    ...     v = mkvc(a, ii)
    ...     print('Shape of output with n_dim =', ii, ': ', v.shape)
    Shape of output with n_dim = 1 :  (6,)
    Shape of output with n_dim = 2 :  (6, 1)
    Shape of output with n_dim = 3 :  (6, 1, 1)
    """
    if isinstance(x, np.matrix):
        x = np.array(x)

    if hasattr(x, 'tovec'):
        x = x.tovec()

    assert isinstance(x, np.ndarray), "Vector must be a numpy array"

    if n_dims == 1:
        return x.flatten(order='F')
    elif n_dims == 2:
        return x.flatten(order='F')[:, np.newaxis]
    elif n_dims == 3:
        return x.flatten(order='F')[:, np.newaxis, np.newaxis]


def ndgrid(*args, **kwargs):
    """
    Form tensorial grid for 1, 2, or 3 dimensions.

    Returns as column vectors by default.

    To return as matrix input:

        ndgrid(..., vector=False)

    The inputs can be a list or separate arguments.

    e.g.::

        a = np.array([1, 2, 3])
        b = np.array([1, 2])

        XY = ndgrid(a, b)
            > [[1 1]
               [2 1]
               [3 1]
               [1 2]
               [2 2]
               [3 2]]

        X, Y = ndgrid(a, b, vector=False)
            > X = [[1 1]
                   [2 2]
                   [3 3]]
            > Y = [[1 2]
                   [1 2]
                   [1 2]]

    """

    # Read the keyword arguments, and only accept a vector=True/False
    vector = kwargs.pop('vector', True)
    assert isinstance(vector, bool), "'vector' keyword must be a bool"
    assert len(kwargs) == 0, "Only 'vector' keyword accepted"

    # you can either pass a list [x1, x2, x3] or each seperately
    if type(args[0]) == list:
        xin = args[0]
    else:
        xin = args

    # Each vector needs to be a numpy array
    assert np.all(
        [isinstance(x, np.ndarray) for x in xin]
    ), "All vectors must be numpy arrays."

    if len(xin) == 1:
        return xin[0]
    elif len(xin) == 2:
        XY = np.broadcast_arrays(mkvc(xin[1], 1), mkvc(xin[0], 2))
        if vector:
            X2, X1 = [mkvc(x) for x in XY]
            return np.c_[X1, X2]
        else:
            return XY[1], XY[0]
    elif len(xin) == 3:
        XYZ = np.broadcast_arrays(
            mkvc(xin[2], 1), mkvc(xin[1], 2), mkvc(xin[0], 3)
        )
        if vector:
            X3, X2, X1 = [mkvc(x) for x in XYZ]
            return np.c_[X1, X2, X3]
        else:
            return XYZ[2], XYZ[1], XYZ[0]
