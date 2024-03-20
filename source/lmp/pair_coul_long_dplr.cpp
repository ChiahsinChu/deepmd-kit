// SPDX-License-Identifier: LGPL-3.0-or-later
#include "pair_coul_long_dplr.h"

#include <cmath>
#include <cstring>

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

#define EWALD_F 1.12837917
#define EWALD_P 0.3275911
#define A1 0.254829592
#define A2 -0.284496736
#define A3 1.421413741
#define A4 -1.453152027
#define A5 1.061405429

/* ---------------------------------------------------------------------- */

PairCoulLongDPLR::PairCoulLongDPLR(LAMMPS *lmp) : PairCoulLong(lmp) {
  // ewaldflag = pppmflag = 1;
  // ftable = nullptr;
  // qdist = 0.0;
  // cut_respa = nullptr;
}

/* ---------------------------------------------------------------------- */

PairCoulLongDPLR::~PairCoulLongDPLR() {
  // if (copymode) {
  //   return;
  // }

  // if (allocated) {
  //   memory->destroy(setflag);
  //   memory->destroy(cutsq);

  //   memory->destroy(scale);
  // }
  // if (ftable) {
  //   free_tables();
  // }
}

/* ---------------------------------------------------------------------- */

void PairCoulLongDPLR::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itable, itype, jtype;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, ecoul, fpair;
  double fraction, table;
  double r, r2inv, forcecoul, factor_coul;
  double grij, expm2, prefactor, t, erfc;
  int *ilist, *jlist, *numneigh, **firstneigh;
  double rsq;

  ecoul = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  fele.resize(static_cast<size_t>(nlocal) * 3);
  fill(fele.begin(), fele.end(), 0.0);

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = type[j];
      //   continue if the atom type is in the atype vector
      if (std::find(atype.begin(), atype.end(), jtype) == atype.end()) {
        continue;
      }

      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_coulsq) {
        r2inv = 1.0 / rsq;
        if (!ncoultablebits || rsq <= tabinnersq) {
          r = sqrt(rsq);
          grij = g_ewald * r;
          expm2 = exp(-grij * grij);
          t = 1.0 / (1.0 + EWALD_P * grij);
          erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
          prefactor = qqrd2e * scale[itype][jtype] * qtmp * q[j] / r;
          forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
          if (factor_coul < 1.0) {
            forcecoul -= (1.0 - factor_coul) * prefactor;
          }
        } else {
          union_int_float_t rsq_lookup;
          rsq_lookup.f = rsq;
          itable = rsq_lookup.i & ncoulmask;
          itable >>= ncoulshiftbits;
          fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
          table = ftable[itable] + fraction * dftable[itable];
          forcecoul = scale[itype][jtype] * qtmp * q[j] * table;
          if (factor_coul < 1.0) {
            table = ctable[itable] + fraction * dctable[itable];
            prefactor = scale[itype][jtype] * qtmp * q[j] * table;
            forcecoul -= (1.0 - factor_coul) * prefactor;
          }
        }

        fpair = forcecoul * r2inv;
        fele[i * 3 + 0] += delx * fpair;
        fele[i * 3 + 1] += dely * fpair;
        fele[i * 3 + 2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          fele[j * 3 + 0] -= delx * fpair;
          fele[j * 3 + 1] -= dely * fpair;
          fele[j * 3 + 2] -= delz * fpair;
        }

        if (eflag) {
          if (!ncoultablebits || rsq <= tabinnersq) {
            ecoul = prefactor * erfc;
          } else {
            table = etable[itable] + fraction * detable[itable];
            ecoul = scale[itype][jtype] * qtmp * q[j] * table;
          }
          if (factor_coul < 1.0) {
            ecoul -= (1.0 - factor_coul) * prefactor;
          }
        }

        if (evflag) {
          ev_tally(i, j, nlocal, newton_pair, 0.0, ecoul, fpair, delx, dely,
                   delz);
        }
      }
    }
  }

  if (vflag_fdotr) {
    virial_fdotr_compute();
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCoulLongDPLR::settings(int narg, char **arg) {
  if (narg <= 1) {
    error->all(FLERR, "Illegal pair_style command");
  }
  cut_coul = utils::numeric(FLERR, arg[0], false, lmp);
  // read atype for calculation until the end of the arg
  // and append to a vector
  for (int i = 1; i < narg; i++) {
    atype.push_back(atoi(arg[i]));
  }

  // std::cout << "cut_coul: " << cut_coul << std::endl;
  // std::cout << "atype: ";
  // for (auto i : atype) {
  //   std::cout << i << " ";
  // }
  // std::cout << std::endl;
}

void PairCoulLongDPLR::init_style() {
  PairCoulLong::init_style();

  int nlocal = atom->nlocal;
  fele.resize(static_cast<size_t>(nlocal) * 3);
  fill(fele.begin(), fele.end(), 0.0);
}

void *PairCoulLongDPLR::extract(const char *str, int &dim)
{
  if (strcmp(str, "cut_coul") == 0) {
    dim = 0;
    return (void *) &cut_coul;
  }
  if (strcmp(str, "scale") == 0) {
    dim = 2;
    return (void *) scale;
  }
  return nullptr;
}

