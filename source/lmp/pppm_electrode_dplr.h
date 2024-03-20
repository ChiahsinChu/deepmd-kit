// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/electrode/dplr, PPPMElectrodeDPLR)
// clang-format on
#else

#ifndef LMP_PPPM_ELECTRODE_DPLR_H
#define LMP_PPPM_ELECTRODE_DPLR_H

#define FLOAT_PREC double

#include <iostream>
#include <vector>

#include "pppm_electrode.h"

namespace LAMMPS_NS {

class PPPMElectrodeDPLR : public PPPMElectrode {
 public:
  PPPMElectrodeDPLR(class LAMMPS *);
  ~PPPMElectrodeDPLR() override{};
  void init() override;
  const std::vector<double> &get_fele() const { return fele; };

 protected:
  // void compute(int, int) override;
  void fieldforce_ik() override;
  void fieldforce_ad() override;

 private:
  std::vector<double> fele;
};

}  // namespace LAMMPS_NS

#endif
#endif
