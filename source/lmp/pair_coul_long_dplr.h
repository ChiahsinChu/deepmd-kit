// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef PAIR_CLASS
// clang-format off
PairStyle(coul/long/dplr,PairCoulLongDPLR);
// clang-format on
#else

#ifndef LMP_PAIR_COUL_LONG_DPLR_H
#define LMP_PAIR_COUL_LONG_DPLR_H

#include "pair_coul_long.h"

namespace LAMMPS_NS {

class PairCoulLongDPLR : public PairCoulLong {
    public:
        PairCoulLongDPLR(class LAMMPS *);
        ~PairCoulLongDPLR() override;
        void compute(int, int) override;
        void settings(int, char **) override;
        void init_style() override;
        void *extract(const char *, int &) override;
        const std::vector<double> &get_fele() const { return fele; };

    protected:
        double cut_coul, cut_coulsq, qdist;
        // double *cut_respa;
        // double g_ewald;
        double **scale;

        std::vector<int> atype;
        std::vector<double> fele;
};

}  // namespace LAMMPS_NS

#endif
#endif
