// SPDX-License-Identifier: LGPL-3.0-or-later
// The ELECTRODE package is only available from LAMMPS 4 May 2022 on, and the
// copy of PPPMElectrode::compute() below uses the Grid3d communication API that
// replaced GridComm in LAMMPS 22 Dec 2022.
#if LAMMPS_VERSION_NUMBER >= 20221222
#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/electrode/dplr, PPPMElectrodeDPLR)
// clang-format on
#else

#ifndef LMP_PPPM_ELECTRODE_DPLR_H
#define LMP_PPPM_ELECTRODE_DPLR_H

#define FLOAT_PREC double

#include <vector>

#include "pppm_electrode.h"

namespace LAMMPS_NS {

/* ----------------------------------------------------------------------
   pppm/electrode with the DPLR force hand-off.

   PPPMElectrode, like plain PPPM, adds the k-space electrostatic force
   straight into atom->f.  That is wrong for DPLR: part of the charge sits on
   massless Wannier centroids, and the force they feel has to be
   back-propagated onto their host atoms by the DipoleChargeModifier.  Both
   fieldforce_* variants are therefore overridden to stash the force in `fele`
   instead, where fix dplr picks it up via get_fele().

   This is to pppm/electrode what pppm/dplr is to pppm.
------------------------------------------------------------------------- */

class PPPMElectrodeDPLR : public PPPMElectrode {
 public:
  PPPMElectrodeDPLR(class LAMMPS *);
  ~PPPMElectrodeDPLR() override {};
  void init() override;
  void compute(int, int) override;
  const std::vector<double> &get_fele() const { return fele; };

 protected:
  void fieldforce_ik() override;
  void fieldforce_ad() override;

 private:
  std::vector<double> fele;
};

}  // namespace LAMMPS_NS

#endif
#endif
#endif  // LAMMPS_VERSION_NUMBER >= 20221222
