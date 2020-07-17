# -*- coding: utf-8 -*-

import numpy as np

from emcee.moves.red_blue import RedBlueMove

__all__ = ["StretchMove"]


class StretchMove(RedBlueMove):
    """
    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.
    :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)
    """

    def __init__(self, a=2.0, n=0, **kwargs):
        self.a = a
        self.n = n
        super(StretchMove, self).__init__(**kwargs)

    def setup(self, coords):
        self.a0 = self.a * 1.3 ** self.n

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        Ns, Nc = len(s), len(c)
        ndim = s.shape[1]
        zz = ((self.a0 - 1.0) * random.rand(Ns) + 1) ** 2.0 / self.a0
        factors = (ndim - 1.0) * np.log(zz)
        rint = random.randint(Nc, size=(Ns,))
        return c[rint] - (c[rint] - s) * zz[:, None], factors
