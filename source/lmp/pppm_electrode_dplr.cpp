// SPDX-License-Identifier: LGPL-3.0-or-later
#include "pppm_electrode_dplr.h"
#include "pppm_electrode.h"

#include <math.h>

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#if LAMMPS_VERSION_NUMBER >= 20221222
#include "grid3d.h"
#else
#include "gridcomm.h"
#endif
#include "math_const.h"
#include "memory.h"
#include "pppm.h"

using namespace LAMMPS_NS;
using namespace MathConst;

enum { REVERSE_RHO };
enum { FORWARD_IK, FORWARD_AD, FORWARD_IK_PERATOM, FORWARD_AD_PERATOM };

#define OFFSET 16384

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF 1.0f
#else
#define ZEROF 0.0
#define ONEF 1.0
#endif

/* ---------------------------------------------------------------------- */


PPPMElectrodeDPLR::PPPMElectrodeDPLR(LAMMPS *lmp) : PPPMElectrode(lmp)
{
  triclinic_support = 1;
}

/* ---------------------------------------------------------------------- */

void PPPMElectrodeDPLR::init() {
  // DPLR PPPM requires newton on, b/c it computes forces on ghost atoms

  if (force->newton == 0) {
    error->all(FLERR, "Kspace style pppm/electrode/dplr requires newton on");
  }

  PPPMElectrode::init();

  int nlocal = atom->nlocal;
  // cout << " ninit pppm/dplr ---------------------- " << nlocal << endl;
  fele.resize(static_cast<size_t>(nlocal) * 3);
  fill(fele.begin(), fele.end(), 0.0);
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
  // double **f = atom->f;

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

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
  // double **f = atom->f;

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

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
