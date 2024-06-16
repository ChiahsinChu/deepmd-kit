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
        self.fitting_setup = kwargs.pop("fitting_setup", {})
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
