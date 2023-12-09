// SPDX-License-Identifier: LGPL-3.0-or-later
#include "pppm_dplr_gpu.h"

#include <math.h>

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#if LAMMPS_VERSION_NUMBER >= 20221222
#include "grid3d.h"
#else
#include "gridcomm.h"
#endif
#include "math_const.h"
#include "memory.h"
#include "neighbor.h"
#include "pppm_gpu.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define MAXORDER 7
#define OFFSET 16384
#define SMALL 0.00001
#define LARGE 10000.0
#define EPS_HOC 1.0e-7

enum{REVERSE_RHO_GPU,REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

// external functions from cuda library for atom decomposition

#ifdef FFT_SINGLE
#define PPPM_GPU_API(api)  pppm_gpu_ ## api ## _f
#else
#define PPPM_GPU_API(api)  pppm_gpu_ ## api ## _d
#endif

FFT_SCALAR* PPPM_GPU_API(init)(const int nlocal, const int nall, FILE *screen,
                               const int order, const int nxlo_out,
                               const int nylo_out, const int nzlo_out,
                               const int nxhi_out, const int nyhi_out,
                               const int nzhi_out, FFT_SCALAR **rho_coeff,
                               FFT_SCALAR **_vd_brick,
                               const double slab_volfactor,
                               const int nx_pppm, const int ny_pppm,
                               const int nz_pppm, const bool split,
                               const bool respa, int &success);
void PPPM_GPU_API(clear)(const double poisson_time);
int PPPM_GPU_API(spread)(const int ago, const int nlocal, const int nall,
                         double **host_x, int *host_type, bool &success,
                         double *host_q, double *boxlo, const double delxinv,
                         const double delyinv, const double delzinv);
void PPPM_GPU_API(interp)(const FFT_SCALAR qqrd2e_scale);
double PPPM_GPU_API(bytes)();
void PPPM_GPU_API(forces)(double **f);

/* ---------------------------------------------------------------------- */

PPPMDPLRGPU::PPPMDPLRGPU(LAMMPS *lmp) : PPPMGPU(lmp)
{
  triclinic_support = 1;
}

/* ---------------------------------------------------------------------- */

void PPPMDPLRGPU::init() {
  // DPLR PPPM requires newton on, b/c it computes forces on ghost atoms

  if (force->newton == 0) {
    error->all(FLERR, "Kspace style pppm/dplr/gpu requires newton on");
  }

  PPPMGPU::init();

  int nlocal = atom->nlocal;
  // cout << " ninit pppm/dplr ---------------------- " << nlocal << endl;
  fele.resize(nlocal * 3);
  fill(fele.begin(), fele.end(), 0.0);
}

/* ----------------------------------------------------------------------
   compute the PPPM long-range force, energy, virial
------------------------------------------------------------------------- */

void PPPMDPLRGPU::compute(int eflag, int vflag) {
  int i,j;

  int nago;
  if (kspace_split) {
    if (im_real_space) return;
    if (atom->nlocal > old_nlocal) {
      nago=0;
      old_nlocal = atom->nlocal;
    } else nago = 1;
  } else nago = neighbor->ago;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  ev_init(eflag,vflag);

  // If need per-atom energies/virials, allocate per-atom arrays here
  // so that particle map on host can be done concurrently with GPU calculations

  if (evflag_atom && !peratom_allocate_flag) {
    allocate_peratom();
  }
   
  if (triclinic == 0) {
    bool success = true;
    int flag=PPPM_GPU_API(spread)(nago, atom->nlocal, atom->nlocal +
                                  atom->nghost, atom->x, atom->type, success,
                                  atom->q, domain->boxlo, delxinv, delyinv,
                                  delzinv);
    if (!success)
      error->one(FLERR,"Insufficient memory on accelerator");
    if (flag != 0)
      error->one(FLERR,"Out of range atoms - cannot compute PPPM");
  }

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }

  // If need per-atom energies/virials, also do particle map on host
  // concurrently with GPU calculations,
  // or if the box is triclinic, particle map is done on host

  if (evflag_atom || triclinic) {

    // extend size of per-atom arrays if necessary

    if (atom->nmax > nmax) {
      memory->destroy(part2grid);
      nmax = atom->nmax;
      memory->create(part2grid,nmax,3,"pppm:part2grid");
    }

    particle_map();
  }

  // if the box is triclinic,
  // map my particle charge onto my local 3d density grid on the host

  if (triclinic) make_rho();

  double t3 = platform::walltime();

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  if (triclinic == 0) {
    // TODO: check version dependency
    gc->reverse_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                     REVERSE_RHO_GPU,gc_buf1,gc_buf2,MPI_FFT_SCALAR);
    brick2fft_gpu();
  } else {
#if LAMMPS_VERSION_NUMBER >= 20221222
    gc->reverse_comm(Grid3d::KSPACE, this, REVERSE_RHO, 1, sizeof(FFT_SCALAR),
                    gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#elif LAMMPS_VERSION_NUMBER >= 20210831 && LAMMPS_VERSION_NUMBER < 20221222
    gc->reverse_comm(GridComm::KSPACE, this, 1, sizeof(FFT_SCALAR), REVERSE_RHO,
                    gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#else
    gc->reverse_comm_kspace(this, 1, sizeof(FFT_SCALAR), REVERSE_RHO, gc_buf1,
                            gc_buf2, MPI_FFT_SCALAR);
#endif
    brick2fft();
  }

  // compute potential gradient on my FFT grid and
  //   portion of e_long on this proc's FFT grid
  // return gradients (electric fields) in 3d brick decomposition

  poisson();

  // all procs communicate E-field values
  // to fill ghost cells surrounding their 3d bricks

  if (differentiation_flag == 1)
#if LAMMPS_VERSION_NUMBER >= 20221222
    gc->reverse_comm(Grid3d::KSPACE, this, REVERSE_RHO, 1, sizeof(FFT_SCALAR),
                     gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#elif LAMMPS_VERSION_NUMBER >= 20210831 && LAMMPS_VERSION_NUMBER < 20221222
    gc->forward_comm(GridComm::KSPACE, this, 1, sizeof(FFT_SCALAR), FORWARD_AD,
                     gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#else
    gc->forward_comm_kspace(this, 1, sizeof(FFT_SCALAR), FORWARD_AD, gc_buf1,
                            gc_buf2, MPI_FFT_SCALAR);
#endif
  else
#if LAMMPS_VERSION_NUMBER >= 20221222
    gc->forward_comm(Grid3d::KSPACE, this, FORWARD_IK, 3, sizeof(FFT_SCALAR),
                     gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#elif LAMMPS_VERSION_NUMBER >= 20210831 && LAMMPS_VERSION_NUMBER < 20221222
    gc->forward_comm(GridComm::KSPACE, this, 3, sizeof(FFT_SCALAR), FORWARD_IK,
                     gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#else
    gc->forward_comm_kspace(this, 3, sizeof(FFT_SCALAR), FORWARD_IK, gc_buf1,
                            gc_buf2, MPI_FFT_SCALAR);
#endif

  // extra per-atom energy/virial communication

  if (evflag_atom) {
    if (differentiation_flag == 1 && vflag_atom)
#if LAMMPS_VERSION_NUMBER >= 20221222
      gc->forward_comm(Grid3d::KSPACE, this, FORWARD_AD_PERATOM, 6,
                       sizeof(FFT_SCALAR), gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#elif LAMMPS_VERSION_NUMBER >= 20210831 && LAMMPS_VERSION_NUMBER < 20221222
      gc->forward_comm(GridComm::KSPACE, this, 6, sizeof(FFT_SCALAR),
                       FORWARD_AD_PERATOM, gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#else
      gc->forward_comm_kspace(this, 6, sizeof(FFT_SCALAR), FORWARD_AD_PERATOM,
                              gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#endif
    else if (differentiation_flag == 0)
#if LAMMPS_VERSION_NUMBER >= 20221222
      gc->forward_comm(Grid3d::KSPACE, this, FORWARD_IK_PERATOM, 7,
                       sizeof(FFT_SCALAR), gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#elif LAMMPS_VERSION_NUMBER >= 20210831 && LAMMPS_VERSION_NUMBER < 20221222
      gc->forward_comm(GridComm::KSPACE, this, 7, sizeof(FFT_SCALAR),
                       FORWARD_IK_PERATOM, gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#else
      gc->forward_comm_kspace(this, 7, sizeof(FFT_SCALAR), FORWARD_IK_PERATOM,
                              gc_buf1, gc_buf2, MPI_FFT_SCALAR);
#endif
  }

  poisson_time += platform::walltime()-t3;

  // calculate the force on my particles
  int nlocal = atom->nlocal;
  double **f = atom->f;
  std::vector<double> tmp_f(nlocal * 3, 0.0);
  for (i = 0; i < nlocal; i++) {
    for (j = 0; j < 3; j++) {
      tmp_f[i * 3 + j] = f[i][j];
    }
  }
  FFT_SCALAR qscale = force->qqrd2e * scale;
  if (triclinic == 0) PPPM_GPU_API(interp)(qscale);
  else fieldforce();
  for (i = 0; i < nlocal; i++) {
    for (j = 0; j < 3; j++) {
      fele[i * 3 + j] = f[i][j] - tmp_f[i * 3 + j];
      f[i][j] = tmp_f[i * 3 + j];
    }
  }

  // per-atom energy/virial
  // energy includes self-energy correction

  if (evflag_atom) fieldforce_peratom();

  // update qsum and qsqsum, if atom count has changed and energy needed

  if ((eflag_global || eflag_atom) && atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  // sum energy across procs and add in volume-dependent term

  if (eflag_global) {
    double energy_all;
    MPI_Allreduce(&energy,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
    energy = energy_all;

    energy *= 0.5*volume;
    energy -= g_ewald*qsqsum/MY_PIS +
      MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy *= qscale;
  }

  // sum virial across procs

  if (vflag_global) {
    double virial_all[6];
    MPI_Allreduce(virial,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    for (i = 0; i < 6; i++) virial[i] = 0.5*qscale*volume*virial_all[i];
  }

  // per-atom energy/virial
  // energy includes self-energy correction

  if (evflag_atom) {
    double *q = atom->q;
    

    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        eatom[i] *= 0.5;
        eatom[i] -= g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /
          (g_ewald*g_ewald*volume);
        eatom[i] *= qscale;
      }
    }

    if (vflag_atom) {
      for (i = 0; i < nlocal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= 0.5*qscale;
    }
  }

  // 2d slab correction

  if (slabflag) slabcorr();

  // convert atoms back from lamda to box coords

  if (triclinic) domain->lamda2x(atom->nlocal);

  if (kspace_split) PPPM_GPU_API(forces)(atom->f);
}
