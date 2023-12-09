// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/dplr/gpu, PPPMDPLRGPU)
// clang-format on
#else

#ifndef LMP_PPPM_DPLR_GPU_H
#define LMP_PPPM_DPLR_GPU_H

#define FLOAT_PREC double

#include <iostream>
#include <vector>

#include "pppm_gpu.h"

namespace LAMMPS_NS {

class PPPMDPLRGPU : public PPPMGPU {
 public:
  PPPMDPLRGPU(class LAMMPS *);
  ~PPPMDPLRGPU() override{};
  void init() override;
  const std::vector<double> &get_fele() const { return fele; };

 protected:
  void compute(int, int) override;
 private:
  std::vector<double> fele;
};

}  // namespace LAMMPS_NS

#endif
#endif
