// SPDX-License-Identifier: LGPL-3.0-or-later
/**
 * See https://docs.lammps.org/Developer_plugins.html
 */
#include "compute_deeptensor_atom.h"
#include "deepmd_version.h"
#include "fix_dplr.h"
#include "lammpsplugin.h"
#include "pair_deepmd.h"
#include "version.h"
#if LAMMPS_VERSION_NUMBER >= 20220328
#include "pppm_dplr.h"
#include "pppm_electrode_dplr.h"
#include "pair_coul_long_dplr.h"
#endif

using namespace LAMMPS_NS;

static Pair *pairdeepmd(LAMMPS *lmp) { return new PairDeepMD(lmp); }

static Compute *computedeepmdtensoratom(LAMMPS *lmp, int narg, char **arg) {
  return new ComputeDeeptensorAtom(lmp, narg, arg);
}

static Fix *fixdplr(LAMMPS *lmp, int narg, char **arg) {
  return new FixDPLR(lmp, narg, arg);
}

#if LAMMPS_VERSION_NUMBER >= 20220328
static KSpace *pppmdplr(LAMMPS *lmp) { return new PPPMDPLR(lmp); }
static KSpace *pppmelectrodedplr(LAMMPS *lmp) { return new PPPMElectrodeDPLR(lmp); }
static Pair *paircoullongdplr(LAMMPS *lmp) { return new PairCoulLongDPLR(lmp); }
#endif

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "deepmd";
  plugin.info = "deepmd pair style " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v1 = (lammpsplugin_factory1 *)&pairdeepmd;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "compute";
  plugin.name = "deeptensor/atom";
  plugin.info = "compute deeptensor/atom " STR_GIT_SUMM;
  plugin.creator.v2 = (lammpsplugin_factory2 *)&computedeepmdtensoratom;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "fix";
  plugin.name = "dplr";
  plugin.info = "fix dplr " STR_GIT_SUMM;
  plugin.creator.v2 = (lammpsplugin_factory2 *)&fixdplr;
  (*register_plugin)(&plugin, lmp);

#if LAMMPS_VERSION_NUMBER >= 20220328
  // lammps/lammps#
  plugin.style = "kspace";
  plugin.name = "pppm/dplr";
  plugin.info = "kspace pppm/dplr " STR_GIT_SUMM;
  plugin.creator.v1 = (lammpsplugin_factory1 *)&pppmdplr;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "kspace";
  plugin.name = "pppm/electrode/dplr";
  plugin.info = "kspace pppm/electrode/dplr " STR_GIT_SUMM;
  plugin.creator.v1 = (lammpsplugin_factory1 *)&pppmelectrodedplr;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "pair";
  plugin.name = "coul/long/dplr";
  plugin.info = "pair coul/long/dplr " STR_GIT_SUMM;
  plugin.creator.v1 = (lammpsplugin_factory1 *)&paircoullongdplr;
  (*register_plugin)(&plugin, lmp);
#endif
}
