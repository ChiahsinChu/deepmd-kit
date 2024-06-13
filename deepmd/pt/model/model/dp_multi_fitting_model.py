# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPMultiFittingAtomicModel,
)
from deepmd.pt.model.model.model import (
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
        assert self.get_fitting_net() is not None

    @torch.jit.export
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
        nlist_reshape = torch.reshape(nlist, [nframes, nloc * nsel, 1])
        mask = nlist_reshape.ge(0)

        ii = torch.arange(nloc, dtype=torch.int64, device=nlist.device)
        ii = torch.tile(ii.reshape(-1, 1), [1, nsel])
        ii = torch.reshape(ii, [nframes, nloc * nsel, 1])
        sel_ii = torch.masked_select(ii, mask)
        sel_ii = torch.reshape(sel_ii, [nframes, -1, 1])

        # nf x (nloc x nsel)
        sel_nlist = torch.masked_select(nlist_reshape, mask)
        sel_jj = torch.gather(mapping, 1, sel_nlist.reshape(nframes, -1))
        sel_jj = torch.reshape(sel_jj, [nframes, -1, 1])

        # nframes x (nloc x nsel) x 3
        pairs = torch.zeros(
            nframes, nloc * nsel, 1, dtype=torch.int64, device=nlist.device
        )
        pairs = torch.masked_select(pairs, mask)
        pairs = torch.reshape(pairs, [nframes, -1, 1])

        pairs = torch.concat([sel_ii, sel_jj, pairs], -1)

        # select the pair with jj > ii
        mask = pairs[..., 1] > pairs[..., 0]
        pairs = torch.masked_select(pairs, mask.reshape(nframes, -1, 1))
        pairs = torch.reshape(pairs, [nframes, -1, 3])
        # todo: padding pairs for jit
        return pairs
