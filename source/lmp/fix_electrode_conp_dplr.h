// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef FIX_CLASS

// clang-format off
FixStyle(electrode/conp/dplr, FixElectrodeConpDPLR);
// clang-format on

#else

#ifndef LMP_FIX_ELECTRODE_CONP_DPLR_H
#define LMP_FIX_ELECTRODE_CONP_DPLR_H

#include "fix_electrode_conp.h"

namespace LAMMPS_NS {

class FixElectrodeConpDPLR : public FixElectrodeConp {
 public:
  FixElectrodeConpDPLR(class LAMMPS *, int, char **);
  // turn off fix efield
  void post_constructor() override;    
  // turn off gausscorr on force
  void setup_pre_reverse(int, int) override; 
  void pre_reverse(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
