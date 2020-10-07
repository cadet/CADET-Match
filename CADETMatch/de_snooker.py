# -*- coding: utf-8 -*-

import numpy as np

from emcee.moves.red_blue import RedBlueMove
import multiprocessing

__all__ = ["DESnookerMove"]


class DESnookerMove(RedBlueMove):
    """A snooker proposal using differential evolution.

    Based on `Ter Braak & Vrugt (2008)
    <http://link.springer.com/article/10.1007/s11222-008-9104-9>`_.

    Credit goes to GitHub user `mdanthony17 <https://github.com/mdanthony17>`_
    for proposing this as an addition to the original emcee package.

    Args:
        gammas (Optional[float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`1.7` as recommended by the
            reference.

    """

    def __init__(self, gammas=1.7, **kwargs):
        self.gammas = gammas
        kwargs["nsplits"] = 4
        super(DESnookerMove, self).__init__(**kwargs)

    def get_proposal(self, s, c, random):
        Ns = len(s)
        Nc = list(map(len, c))
        ndim = s.shape[1]
        q = np.empty((Ns, ndim), dtype=np.float64)
        metropolis = np.empty(Ns, dtype=np.float64)
        for i in range(Ns):
            w = np.array([c[j][random.randint(Nc[j])] for j in range(3)])
            random.shuffle(w)
            z, z1, z2 = w
            delta = s[i] - z
            norm = np.linalg.norm(delta)
            u = delta / np.sqrt(norm)
            q[i] = s[i] + u * self.gammas * (np.dot(u, z1) - np.dot(u, z2))
            q[i] = q[i] % 1
            metropolis[i] = np.log(np.linalg.norm(q[i] - z)) - np.log(norm)

            #if np.any(np.isnan(q[i])):
            #    multiprocessing.get_logger().info("de_snooker q[%s]=%s  w %s  s %s  z %s  z1 %s  z2 %s  delta %s  norm %s  u %s  dot1 %s  dot2 %s  gammas %s", i, q[i], w, s[i],
            #                                      z, z1, z2, delta, norm, u, np.dot(u, z1), np.dot(u, z2), self.gammas)

        #multiprocessing.get_logger().info("de_snooker q %s  s %s  c %s", q, s, c)
        return q, 0.5 * (ndim - 1.0) * metropolis
