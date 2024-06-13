# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np

from deepmd.dpmodel.atomic_model import (
    DPMultiFittingAtomicModel,
)
from deepmd.dpmodel.descriptor import (
    DescrptDPA1,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)


class TestIntegration(unittest.TestCase):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt):
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        self.sel = [5, 2]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut_smth = 0.4
        self.rcut = 2.2

        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        )
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        )
        ft_dict = {
            "type": "test_multi_fitting",
            "ener_1": InvarFitting(
                "energy",
                self.nt,
                ds.get_dim_out(),
                1,
                mixed_types=ds.mixed_types(),
            ),
            "ener_2": InvarFitting(
                "energy",
                self.nt,
                ds.get_dim_out(),
                1,
                mixed_types=ds.mixed_types(),
            ),
        }
        type_map = ["foo", "bar"]
        self.md0 = DPMultiFittingAtomicModel(ds, ft_dict, type_map=type_map)
        self.md1 = DPMultiFittingAtomicModel.deserialize(self.md0.serialize())

    def test_self_consistency(self):
        ret0 = self.md0.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        ret1 = self.md1.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        np.testing.assert_allclose(
            ret0["ener_1"],
            ret1["ener_1"],
        )
        np.testing.assert_allclose(
            ret0["ener_2"],
            ret1["ener_2"],
        )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
