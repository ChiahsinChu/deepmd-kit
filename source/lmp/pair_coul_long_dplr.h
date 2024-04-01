// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef PAIR_CLASS
// clang-format off
PairStyle(coul/long/dplr,PairCoulLongDPLR);
// clang-format on
#else

#ifndef LMP_PAIR_COUL_LONG_DPLR_H
#define LMP_PAIR_COUL_LONG_DPLR_H

#include "pair.h"

namespace LAMMPS_NS {

class PairCoulLongDPLR : public Pair {
    public:
        PairCoulLongDPLR(class LAMMPS *);
        ~PairCoulLongDPLR() override;
        void compute(int, int) override;
        void settings(int, char **) override;
        void coeff(int, char **) override;
        void init_style() override;
        double init_one(int, int) override;
        void write_restart(FILE *) override;
        void read_restart(FILE *) override;
        void write_restart_settings(FILE *) override;
        void read_restart_settings(FILE *) override;
        double single(int, int, int, int, double, double, double, double &) override;
        void *extract(const char *, int &) override;
        const std::vector<double> &get_fele() const { return fele; };

    protected:
        double cut_coul, cut_coulsq, qdist;
        double *cut_respa;
        double g_ewald;
        double **scale;

        virtual void allocate();

        std::vector<int> atype;
        std::vector<double> fele;
};

}  // namespace LAMMPS_NS

#endif
#endif
