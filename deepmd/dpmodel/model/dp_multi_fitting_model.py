# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
)

import numpy as np

from deepmd.dpmodel.atomic_model.dp_multi_fitting_atomic_model import (
    DPMultiFittingAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPMultiFittingModel_ = make_model(DPMultiFittingAtomicModel)


@BaseModel.register("multi_fitting")
class DPMultiFittingModel(DPModelCommon, DPMultiFittingModel_):
    model_type = "multi_fitting"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPMultiFittingModel_.__init__(self, *args, **kwargs)

    def get_sel_type(self) -> List[List[int]]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.atomic_model.get_sel_type()

    @staticmethod
    def make_pairs(nlist, mapping):
        """Return the pairs from nlist and mapping.

        Returns
        -------
        pairs
            [[i1, j1, 0], [i2, j2, 0], ...],
            in which i and j are the local indices of the atoms

        """
        nframes, nloc, nsel = nlist.shape
        assert nframes == 1
        nlist_reshape = np.reshape(nlist, [nframes, nloc * nsel])
        # nlist is pad with -1
        mask = nlist_reshape >= 0

        ii = np.arange(nloc, dtype=np.int64)
        ii = np.tile(ii.reshape(-1, 1), [1, nsel])
        ii = np.reshape(ii, [nframes, nloc * nsel])
        # nf x (nloc x nsel)
        sel_ii = ii[mask]
        # sel_ii = np.reshape(sel_ii, [nframes, -1, 1])

        # nf x (nloc x nsel)
        sel_nlist = nlist_reshape[mask]
        sel_jj = np.take_along_axis(mapping, sel_nlist, axis=1)
        sel_jj = np.reshape(sel_jj, [nframes, -1])

        # nframes x (nloc x nsel) x 3
        pairs = np.zeros([nframes, nloc * nsel], dtype=np.int64)
        pairs = np.stack((sel_ii, sel_jj, pairs[mask]))

        # select the pair with jj > ii
        # nframes x (nloc x nsel)
        mask = pairs[..., 1] > pairs[..., 0]
        pairs = pairs[mask]
        pairs = np.reshape(pairs, [nframes, -1, 3])

        return pairs
