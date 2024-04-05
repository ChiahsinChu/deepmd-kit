// SPDX-License-Identifier: LGPL-3.0-or-later
#include "fix_electrode_conp_dplr.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "electrode_accel_interface.h"
#include "electrode_math.h"
#include "electrode_matrix.h"
#include "electrode_vector.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "pointers.h"
#include "text_file_reader.h"
#include "variable.h"

#include <cassert>
#include <numeric>

using namespace LAMMPS_NS;
using namespace MathConst;


FixElectrodeConpDPLR::FixElectrodeConpDPLR(LAMMPS *lmp, int narg, char **arg) :
    FixElectrodeConp(lmp, narg, arg)
{}

/* ---------------------------------------------------------------------- */

void FixElectrodeConpDPLR::post_constructor()
{
  if (!ffield) return;
  // ffield: test conditions and set up efield
  if (num_of_groups != 2) error->all(FLERR, "Number of electrodes must be two with ffield yes");
  if (!symm) error->all(FLERR, "Keyword symm off not allowed with ffield yes");
  if (domain->zperiodic == 0 || domain->boundary[2][0] != 0 || domain->boundary[2][1] != 0)
    error->all(FLERR, "Periodic z boundaries required with ffield yes");

  top_group = get_top_group();
  // assign variable names:
  std::string var_vtop = fixname + "_ffield_vtop";
  std::string var_vbot = fixname + "_ffield_vbot";
  std::string var_efield = fixname + "_ffield_zfield";
  // set variables:
  input->variable->set(fmt::format("{} equal f_{}[{}]", var_vbot, fixname, 1 + 1 - top_group));
  input->variable->set(fmt::format("{} equal f_{}[{}]", var_vtop, fixname, 1 + top_group));
  input->variable->set(fmt::format("{} equal (v_{}-v_{})/lz", var_efield, var_vbot, var_vtop));
  // check for other efields and warn if found
  if (modify->get_fix_by_style("efield").size() > 0)
    error->warning(FLERR, "Other efield fixes found -- please make sure this is intended!");
//   // call fix command:
//   // fix [varstem]_efield all efield 0.0 0.0 [var_vdiff]/lz
//   std::string efield_call = fixname + "_efield all efield 0.0 0.0 v_" + var_efield;
//   modify->add_fix(efield_call, 1);
}

/* ---------------------------------------------------------------------- */

void FixElectrodeConpDPLR::setup_pre_reverse(int eflag, int /*vflag*/)
{
  // correct forces for initial timestep
  gausscorr(eflag, false);
  self_energy(eflag);
  potential_energy(eflag);
}

/* ---------------------------------------------------------------------- */

void FixElectrodeConpDPLR::pre_reverse(int eflag, int /*vflag*/)
{
  gausscorr(eflag, false);
  self_energy(eflag);
  potential_energy(eflag);
}