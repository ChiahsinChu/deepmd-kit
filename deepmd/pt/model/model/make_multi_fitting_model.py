# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

import torch

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableCategory,
    OutputVariableOperation,
    check_operation_applied,
)
from deepmd.pt.model.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_ENER_FLOAT_PRECISION,
    GLOBAL_PT_FLOAT_PRECISION,
    PRECISION_DICT,
    RESERVED_PRECISON_DICT,
)
from deepmd.pt.utils.nlist import (
    nlist_distinguish_types,
)
from deepmd.utils.path import (
    DPPath,
)


def make_multi_fitting_model(T_AtomicModel: Type[BaseAtomicModel]):
    class CM(BaseModel):
        def __init__(
            self,
            *args,
            # underscore to prevent conflict with normal inputs
            atomic_model_: Optional[T_AtomicModel] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if atomic_model_ is not None:
                self.atomic_model: T_AtomicModel = atomic_model_
            else:
                self.atomic_model: T_AtomicModel = T_AtomicModel(*args, **kwargs)
            self.precision_dict = PRECISION_DICT
            self.reverse_precision_dict = RESERVED_PRECISON_DICT
            self.global_pt_float_precision = GLOBAL_PT_FLOAT_PRECISION
            self.global_pt_ener_float_precision = GLOBAL_PT_ENER_FLOAT_PRECISION

        def model_output_def(self):
            """Get the output def for the model."""
            return ModelOutputDef(self.atomic_output_def())

        @torch.jit.export
        def model_output_type(self) -> List[str]:
            """Get the output type for the model."""
            output_def = self.model_output_def()
            var_defs = output_def.var_defs
            # jit: Comprehension ifs are not supported yet
            # type hint is critical for JIT
            vars: List[str] = []
            for kk, vv in var_defs.items():
                # .value is critical for JIT
                if vv.category == OutputVariableCategory.OUT.value:
                    vars.append(kk)
            return vars

        def get_out_bias(self) -> torch.Tensor:
            return self.atomic_model.get_out_bias()

        def change_out_bias(
            self,
            merged,
            bias_adjust_mode="change-by-statistic",
        ) -> None:
            """Change the output bias of atomic model according to the input data and the pretrained model.

            Parameters
            ----------
            merged : Union[Callable[[], List[dict]], List[dict]]
                - List[dict]: A list of data samples from various data systems.
                    Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                    originating from the `i`-th data system.
                - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                    only when needed. Since the sampling process can be slow and memory-intensive,
                    the lazy function helps by only sampling once.
            bias_adjust_mode : str
                The mode for changing output bias : ['change-by-statistic', 'set-by-statistic']
                'change-by-statistic' : perform predictions on labels of target dataset,
                        and do least square on the errors to obtain the target shift as bias.
                'set-by-statistic' : directly use the statistic output bias in the target dataset.
            """
            self.atomic_model.change_out_bias(
                merged,
                bias_adjust_mode=bias_adjust_mode,
            )

        def input_type_cast(
            self,
            coord: torch.Tensor,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            str,
        ]:
            """Cast the input data to global float type."""
            input_prec = self.reverse_precision_dict[coord.dtype]
            ###
            ### type checking would not pass jit, convert to coord prec anyway
            ###
            # for vv, kk in zip([fparam, aparam], ["frame", "atomic"]):
            #     if vv is not None and self.reverse_precision_dict[vv.dtype] != input_prec:
            #         log.warning(
            #           f"type of {kk} parameter {self.reverse_precision_dict[vv.dtype]}"
            #           " does not match"
            #           f" that of the coordinate {input_prec}"
            #         )
            _lst: List[Optional[torch.Tensor]] = [
                vv.to(coord.dtype) if vv is not None else None
                for vv in [box, fparam, aparam]
            ]
            box, fparam, aparam = _lst
            if (
                input_prec
                == self.reverse_precision_dict[self.global_pt_float_precision]
            ):
                return coord, box, fparam, aparam, input_prec
            else:
                pp = self.global_pt_float_precision
                return (
                    coord.to(pp),
                    box.to(pp) if box is not None else None,
                    fparam.to(pp) if fparam is not None else None,
                    aparam.to(pp) if aparam is not None else None,
                    input_prec,
                )

        def output_type_cast(
            self,
            model_ret: Dict[str, torch.Tensor],
            input_prec: str,
        ) -> Dict[str, torch.Tensor]:
            """Convert the model output to the input prec."""
            do_cast = (
                input_prec
                != self.reverse_precision_dict[self.global_pt_float_precision]
            )
            pp = self.precision_dict[input_prec]
            odef = self.model_output_def()
            for kk in odef.keys():
                if kk not in model_ret.keys():
                    # do not return energy_derv_c if not do_atomic_virial
                    continue
                if check_operation_applied(odef[kk], OutputVariableOperation.REDU):
                    model_ret[kk] = (
                        model_ret[kk].to(self.global_pt_ener_float_precision)
                        if model_ret[kk] is not None
                        else None
                    )
                elif do_cast:
                    model_ret[kk] = (
                        model_ret[kk].to(pp) if model_ret[kk] is not None else None
                    )
            return model_ret

        def format_nlist(
            self,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
        ):
            """Format the neighbor list.

            1. If the number of neighbors in the `nlist` is equal to sum(self.sel),
            it does nothong

            2. If the number of neighbors in the `nlist` is smaller than sum(self.sel),
            the `nlist` is pad with -1.

            3. If the number of neighbors in the `nlist` is larger than sum(self.sel),
            the nearest sum(sel) neighbors will be preseved.

            Known limitations:

            In the case of not self.mixed_types, the nlist is always formatted.
            May have side effact on the efficiency.

            Parameters
            ----------
            extended_coord
                coodinates in extended region. nf x nall x 3
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel

            Returns
            -------
            formated_nlist
                the formated nlist.

            """
            mixed_types = self.mixed_types()
            nlist = self._format_nlist(extended_coord, nlist, sum(self.get_sel()))
            if not mixed_types:
                nlist = nlist_distinguish_types(nlist, extended_atype, self.get_sel())
            return nlist

        def _format_nlist(
            self,
            extended_coord: torch.Tensor,
            nlist: torch.Tensor,
            nnei: int,
        ):
            n_nf, n_nloc, n_nnei = nlist.shape
            # nf x nall x 3
            extended_coord = extended_coord.view([n_nf, -1, 3])
            rcut = self.get_rcut()

            if n_nnei < nnei:
                nlist = torch.cat(
                    [
                        nlist,
                        -1
                        * torch.ones(
                            [n_nf, n_nloc, nnei - n_nnei],
                            dtype=nlist.dtype,
                            device=nlist.device,
                        ),
                    ],
                    dim=-1,
                )
            elif n_nnei > nnei:
                m_real_nei = nlist >= 0
                nlist = torch.where(m_real_nei, nlist, 0)
                # nf x nloc x 3
                coord0 = extended_coord[:, :n_nloc, :]
                # nf x (nloc x nnei) x 3
                index = nlist.view(n_nf, n_nloc * n_nnei, 1).expand(-1, -1, 3)
                coord1 = torch.gather(extended_coord, 1, index)
                # nf x nloc x nnei x 3
                coord1 = coord1.view(n_nf, n_nloc, n_nnei, 3)
                # nf x nloc x nnei
                rr = torch.linalg.norm(coord0[:, :, None, :] - coord1, dim=-1)
                rr = torch.where(m_real_nei, rr, float("inf"))
                rr, nlist_mapping = torch.sort(rr, dim=-1)
                nlist = torch.gather(nlist, 2, nlist_mapping)
                nlist = torch.where(rr > rcut, -1, nlist)
                nlist = nlist[..., :nnei]
            else:  # n_nnei == nnei:
                pass  # great!
            assert nlist.shape[-1] == nnei
            return nlist

        def do_grad_r(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is r_differentiable.
            if var_name is None, returns if any of the variable is r_differentiable.
            """
            return self.atomic_model.do_grad_r(var_name)

        def do_grad_c(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is c_differentiable.
            if var_name is None, returns if any of the variable is c_differentiable.
            """
            return self.atomic_model.do_grad_c(var_name)

        def serialize(self) -> dict:
            return self.atomic_model.serialize()

        @classmethod
        def deserialize(cls, data) -> "CM":
            return cls(atomic_model_=T_AtomicModel.deserialize(data))

        @torch.jit.export
        def get_dim_fparam(self) -> int:
            """Get the number (dimension) of frame parameters of this atomic model."""
            return self.atomic_model.get_dim_fparam()

        @torch.jit.export
        def get_dim_aparam(self) -> int:
            """Get the number (dimension) of atomic parameters of this atomic model."""
            return self.atomic_model.get_dim_aparam()

        @torch.jit.export
        def get_sel_type(self) -> List[int]:
            """Get the selected atom types of this model.

            Only atoms with selected atom types have atomic contribution
            to the result of the model.
            If returning an empty list, all atom types are selected.
            """
            return self.atomic_model.get_sel_type()

        @torch.jit.export
        def is_aparam_nall(self) -> bool:
            """Check whether the shape of atomic parameters is (nframes, nall, ndim).

            If False, the shape is (nframes, nloc, ndim).
            """
            return self.atomic_model.is_aparam_nall()

        @torch.jit.export
        def get_rcut(self) -> float:
            """Get the cut-off radius."""
            return self.atomic_model.get_rcut()

        @torch.jit.export
        def get_type_map(self) -> List[str]:
            """Get the type map."""
            return self.atomic_model.get_type_map()

        @torch.jit.export
        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nsel()

        @torch.jit.export
        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nnei()

        def atomic_output_def(self) -> FittingOutputDef:
            """Get the output def of the atomic model."""
            return self.atomic_model.atomic_output_def()

        def compute_or_load_stat(
            self,
            sampled_func,
            stat_file_path: Optional[DPPath] = None,
        ):
            """Compute or load the statistics."""
            return self.atomic_model.compute_or_load_stat(sampled_func, stat_file_path)

        def get_sel(self) -> List[int]:
            """Returns the number of selected atoms for each type."""
            return self.atomic_model.get_sel()

        def mixed_types(self) -> bool:
            """If true, the model
            1. assumes total number of atoms aligned across frames;
            2. uses a neighbor list that does not distinguish different atomic types.

            If false, the model
            1. assumes total number of atoms of each atom type aligned across frames;
            2. uses a neighbor list that distinguishes different atomic types.

            """
            return self.atomic_model.mixed_types()

        @torch.jit.export
        def has_message_passing(self) -> bool:
            """Returns whether the model has message passing."""
            return self.atomic_model.has_message_passing()

        @staticmethod
        def make_pairs(nlist, mapping):
            """
            return the pairs from nlist and mapping
            pairs:
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
            return pairs

    return CM
