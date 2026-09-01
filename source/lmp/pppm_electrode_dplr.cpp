// SPDX-License-Identifier: LGPL-3.0-or-later
#include "pppm_electrode_dplr.h"

#if LAMMPS_VERSION_NUMBER >= 20221222

#include <math.h>

#include "atom.h"
#include "boundary_correction.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "grid3d.h"
#include "math_const.h"
#include "memory.h"
#include "pppm_electrode.h"

using namespace LAMMPS_NS;
using namespace MathConst;

// LAMMPS 10 Oct 2024 moved these into KSpace; where that has happened the
// class-scope enumerators win name lookup inside the member functions below
// and these are simply unused.
enum { REVERSE_RHO };
enum { FORWARD_IK, FORWARD_AD, FORWARD_IK_PERATOM, FORWARD_AD_PERATOM };

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF 1.0f
#else
#define ZEROF 0.0
#define ONEF 1.0
#endif

/* ---------------------------------------------------------------------- */

PPPMElectrodeDPLR::PPPMElectrodeDPLR(LAMMPS *lmp) : PPPMElectrode(lmp) {}

/* ---------------------------------------------------------------------- */

void PPPMElectrodeDPLR::init() {
  // DPLR PPPM requires newton on, b/c it computes forces on ghost atoms

  if (force->newton == 0) {
    error->all(FLERR, "Kspace style pppm/electrode/dplr requires newton on");
  }

  PPPMElectrode::init();

  int nlocal = atom->nlocal;
  fele.resize(static_cast<size_t>(nlocal) * 3);
  fill(fele.begin(), fele.end(), 0.0);
}

/* ----------------------------------------------------------------------
   compute the PPPM long-range force, energy, virial

   Copy of PPPMElectrode::compute() with the same edit pppm/dplr makes to
   pppm: the Ewald self-energy is not subtracted from the global energy, so
   that elong keeps the convention the DPLR model was trained with.  The
   per-atom self term in eatom is left alone, again as in pppm/dplr.
------------------------------------------------------------------------- */

void PPPMElectrodeDPLR::compute(int eflag, int vflag) {
  int i, j;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  ev_init(eflag, vflag);

  if (evflag_atom && !peratom_allocate_flag) {
    allocate_peratom();
  }

  // if atom count has changed, update qsum and qsqsum

  qsum_qsq();
  natoms_original = atom->natoms;

  // PPPMElectrode::start_compute() is private, so do here what it does.
  // Its compute_step guard only serves to skip a repeated particle_map() when
  // compute_vector() already ran this step; repeating it is correct, just
  // redundant.  Its first-call setup() is not needed either: LAMMPS calls
  // kspace->setup() from the Verlet/Min setup before compute() is reached.
  // The compute_vector_called fast path of the base class is skipped for the
  // same reason -- that flag is private too -- which leaves the density being
  // rebuilt from scratch every step, exactly as upstream does unconditionally
  // from LAMMPS 29 Aug 2024 on.

  boxlo = domain->boxlo;

  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy(part2grid);
    nmax = atom->nmax;
    memory->create(part2grid, nmax, 3, "pppm/electrode/dplr:part2grid");
  }

  // find grid points for all my particles
  // map my particle charge onto my local 3d density grid

  particle_map();
  make_rho();

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  gc->reverse_comm(Grid3d::KSPACE, this, REVERSE_RHO, 1, sizeof(FFT_SCALAR),
                   gc_buf1, gc_buf2, MPI_FFT_SCALAR);

  brick2fft();

  // compute potential gradient on my FFT grid and
  //   portion of e_long on this proc's FFT grid
  // return gradients (electric fields) in 3d brick decomposition
  // also performs per-atom calculations via poisson_peratom()

  poisson();

  // all procs communicate E-field values
  // to fill ghost cells surrounding their 3d bricks

  if (differentiation_flag == 1) {
    gc->forward_comm(Grid3d::KSPACE, this, FORWARD_AD, 1, sizeof(FFT_SCALAR),
                     gc_buf1, gc_buf2, MPI_FFT_SCALAR);
  } else {
    gc->forward_comm(Grid3d::KSPACE, this, FORWARD_IK, 3, sizeof(FFT_SCALAR),
                     gc_buf1, gc_buf2, MPI_FFT_SCALAR);
  }

  // extra per-atom energy/virial communication

  if (evflag_atom) {
    if (differentiation_flag == 1 && vflag_atom) {
      gc->forward_comm(Grid3d::KSPACE, this, FORWARD_AD_PERATOM, 6,
                       sizeof(FFT_SCALAR), gc_buf1, gc_buf2, MPI_FFT_SCALAR);
    } else if (differentiation_flag == 0) {
      gc->forward_comm(Grid3d::KSPACE, this, FORWARD_IK_PERATOM, 7,
                       sizeof(FFT_SCALAR), gc_buf1, gc_buf2, MPI_FFT_SCALAR);
    }
  }

  // calculate the force on my particles

  fieldforce();

  // extra per-atom energy/virial communication

  if (evflag_atom) {
    fieldforce_peratom();
  }

  // sum global energy across procs and add in volume-dependent term

  const double qscale = qqrd2e * scale;

  if (eflag_global) {
    double energy_all;
    MPI_Allreduce(&energy, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    energy = energy_all;

    energy *= 0.5 * volume;
    // do not add self-term, for neutral systems qsum == 0
    // energy -= g_ewald*qsqsum/MY_PIS +
    //   MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy *= qscale;
  }

  // sum global virial across procs

  if (vflag_global) {
    double virial_all[6];
    MPI_Allreduce(virial, virial_all, 6, MPI_DOUBLE, MPI_SUM, world);
    for (i = 0; i < 6; i++) {
      virial[i] = 0.5 * qscale * volume * virial_all[i];
    }
  }

  // per-atom energy/virial
  // energy includes self-energy correction
  // ntotal accounts for TIP4P tallying eatom/vatom for ghost atoms

  if (evflag_atom) {
    double *q = atom->q;
    int nlocal = atom->nlocal;
    int ntotal = nlocal;
    if (tip4pflag) {
      ntotal += atom->nghost;
    }

    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        eatom[i] *= 0.5;
        eatom[i] -= g_ewald * q[i] * q[i] / MY_PIS +
                    MY_PI2 * q[i] * qsum / (g_ewald * g_ewald * volume);
        eatom[i] *= qscale;
      }
      for (i = nlocal; i < ntotal; i++) {
        eatom[i] *= 0.5 * qscale;
      }
    }

    if (vflag_atom) {
      for (i = 0; i < ntotal; i++) {
        for (j = 0; j < 6; j++) {
          vatom[i][j] *= 0.5 * qscale;
        }
      }
    }
  }

  // slab/wire boundary correction; replaces PPPM's slabcorr()

  boundcorr->compute_corr(qsum, eflag_atom, eflag_global, energy, eatom);
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void PPPMElectrodeDPLR::fieldforce_ik() {
  int i, l, m, n, nx, ny, nz, mx, my, mz;
  FFT_SCALAR dx, dy, dz, x0, y0, z0;
  FFT_SCALAR ekx, eky, ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;

  int nlocal = atom->nlocal;

  fele.resize(static_cast<size_t>(nlocal) * 3);
  fill(fele.begin(), fele.end(), 0.0);

  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

    compute_rho1d(dx, dy, dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n + nz;
      z0 = rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m + ny;
        y0 = z0 * rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l + nx;
          x0 = y0 * rho1d[0][l];
          ekx -= x0 * vdx_brick[mz][my][mx];
          eky -= x0 * vdy_brick[mz][my][mx];
          ekz -= x0 * vdz_brick[mz][my][mx];
        }
      }
    }

    // convert E-field to force

    const double qfactor = qqrd2e * scale * q[i];
    fele[i * 3 + 0] += qfactor * ekx;
    fele[i * 3 + 1] += qfactor * eky;
    if (slabflag != 2) {
      fele[i * 3 + 2] += qfactor * ekz;
    }
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ad
------------------------------------------------------------------------- */

void PPPMElectrodeDPLR::fieldforce_ad() {
  int i, l, m, n, nx, ny, nz, mx, my, mz;
  FFT_SCALAR dx, dy, dz;
  FFT_SCALAR ekx, eky, ekz;
  double s1, s2, s3;
  double sf = 0.0;
  double *prd;

  prd = domain->prd;
  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  double hx_inv = nx_pppm / xprd;
  double hy_inv = ny_pppm / yprd;
  double hz_inv = nz_pppm / zprd;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;

  int nlocal = atom->nlocal;

  fele.resize(static_cast<size_t>(nlocal) * 3);
  fill(fele.begin(), fele.end(), 0.0);

  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

    compute_rho1d(dx, dy, dz);
    compute_drho1d(dx, dy, dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n + nz;
      for (m = nlower; m <= nupper; m++) {
        my = m + ny;
        for (l = nlower; l <= nupper; l++) {
          mx = l + nx;
          ekx += drho1d[0][l] * rho1d[1][m] * rho1d[2][n] * u_brick[mz][my][mx];
          eky += rho1d[0][l] * drho1d[1][m] * rho1d[2][n] * u_brick[mz][my][mx];
          ekz += rho1d[0][l] * rho1d[1][m] * drho1d[2][n] * u_brick[mz][my][mx];
        }
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;

    // convert E-field to force and subtract self forces

    const double qfactor = qqrd2e * scale;

    s1 = x[i][0] * hx_inv;
    s2 = x[i][1] * hy_inv;
    s3 = x[i][2] * hz_inv;
    sf = sf_coeff[0] * sin(2 * MY_PI * s1);
    sf += sf_coeff[1] * sin(4 * MY_PI * s1);
    sf *= 2 * q[i] * q[i];
    fele[i * 3 + 0] += qfactor * (ekx * q[i] - sf);

    sf = sf_coeff[2] * sin(2 * MY_PI * s2);
    sf += sf_coeff[3] * sin(4 * MY_PI * s2);
    sf *= 2 * q[i] * q[i];
    fele[i * 3 + 1] += qfactor * (eky * q[i] - sf);

    sf = sf_coeff[4] * sin(2 * MY_PI * s3);
    sf += sf_coeff[5] * sin(4 * MY_PI * s3);
    sf *= 2 * q[i] * q[i];
    if (slabflag != 2) {
      fele[i * 3 + 2] += qfactor * (ekz * q[i] - sf);
    }
  }
}

#endif  // LAMMPS_VERSION_NUMBER >= 20221222
