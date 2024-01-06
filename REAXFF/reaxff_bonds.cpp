// clang-format off
/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, hmaktulga@lbl.gov
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  Please cite the related publication:
  H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
  "Parallel Reactive Molecular Dynamics: Numerical Methods and
  Algorithmic Techniques", Parallel Computing, 38 (4-5), 245-259

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <https://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#include "reaxff_api.h"

#include "pair.h"

#include <cmath>

namespace ReaxFF {

  double enn(rvec x, network_parameters *fnp, rvec delta, int m, int n) {
    /***  a neural network function to compute bond-order correction coefficence  ***/
    double zi[m],ai[m];
    double zh[m],ah[n][m];
    double ao,zo,sp_o;
    double delta_i[m][3];
    double delta_h[n][m][m];
    double sp_i[m],sp_h[n][m];  // sigma_prime: the derivative of activation function
    double wh[m][m];

    //fprintf(stderr,"\t x (enn): %f %f %f \n", x[0],x[1],x[2]);
    memset(delta_h, 0, sizeof(delta_h));
//    ****    input layer: zi = x*wi + b    ***  (m,n) = (3,9)
    for (int i = 0; i < m; i++) {
        zi[i] = 0.0;
        for (int j = 0; j < 3; j++) {
            zi[i] = zi[i] + fnp->wi[i][j] * x[j];
        }
        zi[i]   = zi[i] + fnp->bi[i];
        ai[i]   = 1.0/(1.0+exp(-zi[i]));    // output of input layers
        sp_i[i] = ai[i]*(1.0-ai[i]);        // sigma_prime of inpute layer
    }

//    ****    hidden layer: zh = x*wh + b    ***  (9,9)
    if (n>0){
      for (int l = 0; l < n; l++) {
          for (int i = 0; i < m; i++) {
              zh[i] = 0.0;
              for (int j = 0; j < m; j++) {
                  if (l==0) {
                     zh[i] = zh[i] + fnp->wh[l][i][j] * ai[j];
                  } else {
                     zh[i] = zh[i] + fnp->wh[l][i][j] * ah[l-1][j];
                  }
                }
              zh[i] = zh[i] + fnp->bh[l][i];
              ah[l][i] = 1.0/(1.0+exp(-zh[i]));         // output of hidden layers
              sp_h[l][i] = ah[l][i]*(1.0-ah[l][i]);        // sigma_prime of inpute layer
          }
      }
    } else {  // no hidden layer
      for (int i = 0; i < m; i++) { 
        ah[n-1][i] = ai[i];
      }
    } //  end hidden layer

//  ****    output layer: zo = ah*wo + b    ***  (9,1)
    //for (int i = 0; i < 3; i++) {
        zo = 0.0;
        for (int j = 0; j < m; j++) {
            zo = zo + fnp->wo[0][j] * ah[n-1][j];
        }
        zo = zo + fnp->bo[0];
        ao = 1.0/(1.0+exp(-zo));       // output of input layers
        sp_o = ao*(1.0-ao);         // sigma_prime of output layer
    //}

//    ****    derivative of input layer neural networks   ****  
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < m; i++) {
            delta_i[i][j] = sp_i[i]*fnp->wi[i][j];  //! reaxFFwi 6*3 delta_i 6*3
        }
    }

    for (int l = 0; l < n; l++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < m; k++) {
                for (int i = 0; i < m; i++) {
                    if (k == 0) delta_h[l][i][j] = 0.0; 
                    if (l==0) {
                      delta_h[l][i][j] = delta_h[l][i][j] + sp_h[l][i]*fnp->wh[l][i][k]*delta_i[k][j];       // m*3
                    } else { 
                      delta_h[l][i][j] = delta_h[l][i][j] + sp_h[l][i]*fnp->wh[l][i][k]*delta_h[l-1][k][j];  // m*3
                    }
                }
            }
        }
    }

    delta[0] = delta[1] = delta[2] = 0.0;
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < m; k++) {
          //for (int i = 0; i < 3; i++) {
            if (k==0) delta[j] = 0.0;
            if (n>0) {
                delta[j] = delta[j] + sp_o*fnp->wo[0][k]*delta_h[n-1][k][j];      // output layer [i,j]
            } else {
                delta[j] = delta[j] + sp_o*fnp->wo[0][k]*delta_i[k][j];           // output layer [i,j]
            }
          //}
      }
    }
    //fprintf(stderr,"delta: %f %f %f \n", delta[0],delta[1],delta[2]);
    //fprintf(stderr,"ao   : %f \n", ao);
    // exit(0);
    return ao;
  }

  void Bonds(reax_system *system, control_params *control, simulation_data *data, storage *workspace, reax_list **lists)
  {
    int i, j, pj, natoms;
    int start_i, end_i;
    int type_i, type_j;
    double ebond, pow_BOs_be2, exp_be12, CEbo;
    double gp3, gp4, gp7, gp10, gp37;
    double exphu, exphua1, exphub1, exphuov, hulpov, estriph;
    double decobdbo, decobdboua, decobdboub;
    double x[3],fe,dfe[3];                                  /***  used by nn  ***/ 
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    network_parameters  *enp;                                    /***  used by nn  ***/          
    bond_order_data *bo_ij;
    reax_list *bonds;

    bonds = (*lists) + BONDS;
    gp3 = system->reax_param.gp.l[3];
    gp4 = system->reax_param.gp.l[4];
    gp7 = system->reax_param.gp.l[7];
    gp10 = system->reax_param.gp.l[10];
    gp37 = (int) system->reax_param.gp.l[37];
    natoms = system->n;
    //fprintf(stderr,"\t gp37: %f  \n", gp37);
    for (i = 0; i < natoms; ++i) {
      start_i = Start_Index(i, bonds);
      end_i = End_Index(i, bonds);

      for (pj = start_i; pj < end_i; ++pj) {
        j = bonds->select.bond_list[pj].nbr;

        if (system->my_atoms[i].orig_id > system->my_atoms[j].orig_id)
          continue;
        if (system->my_atoms[i].orig_id == system->my_atoms[j].orig_id) {
          if (system->my_atoms[j].x[2] <  system->my_atoms[i].x[2]) continue;
          if (system->my_atoms[j].x[2] == system->my_atoms[i].x[2] &&
              system->my_atoms[j].x[1] <  system->my_atoms[i].x[1]) continue;
          if (system->my_atoms[j].x[2] == system->my_atoms[i].x[2] &&
              system->my_atoms[j].x[1] == system->my_atoms[i].x[1] &&
              system->my_atoms[j].x[0] <  system->my_atoms[i].x[0]) continue;
        }

        /* set the pointers */
        type_i = system->my_atoms[i].type;
        type_j = system->my_atoms[j].type;
        if ((type_i < 0) || (type_j < 0)) continue;
        sbp_i = &(system->reax_param.sbp[type_i]);
        sbp_j = &(system->reax_param.sbp[type_j]);
        twbp = &(system->reax_param.tbp[type_i][type_j]);
        enp  = &(system->reax_param.ennp[type_i][type_j]);                /***  used by nn  ***/
        bo_ij = &(bonds->select.bond_list[pj].bo_data);

        /* calculate the constants */
        if (bo_ij->BO_s == 0.0) pow_BOs_be2 = 0.0;
        else pow_BOs_be2 = pow(bo_ij->BO_s, twbp->p_be2);
        exp_be12 = exp(twbp->p_be1 * (1.0 - pow_BOs_be2));
        CEbo = -twbp->De_s * exp_be12 *
          (1.0 - twbp->p_be1 * twbp->p_be2 * pow_BOs_be2);

        /* calculate the Bond Energy */
        
        if (control->nnflag){
          x[0] = bo_ij->BO_s;
          x[1] = bo_ij->BO_pi;
          x[2] = bo_ij->BO_pi2;
          fe = enn(x, enp, dfe, control->belayer_m, control->belayer_n);  /***  nn energy function  ***/
          data->my_en.e_bond += ebond = -twbp->De_s * fe;
          //fprintf(stderr,"\t ebond (%d, %d) %f \n",type_i,type_j,ebond);
        } else {
          data->my_en.e_bond += ebond =
            -twbp->De_s * bo_ij->BO_s * exp_be12
            -twbp->De_p * bo_ij->BO_pi
            -twbp->De_pp * bo_ij->BO_pi2;
        }
        /* tally energy into global or per-atom energy accumulators */
        if (system->pair_ptr->eflag_either)
          system->pair_ptr->ev_tally(i,j,natoms,1,ebond,0.0,0.0,0.0,0.0,0.0);

        /* calculate derivatives of Bond Orders */
        if (control->nnflag){
          bo_ij->Cdbos -= dfe[0]*twbp->De_s;
          bo_ij->Cdbopi -= dfe[1]*twbp->De_s;
          bo_ij->Cdbopi2 -= dfe[2]*twbp->De_s;
          // bo_ij->Cdbo  =  bo_ij->Cdbos + bo_ij->Cdbopi + bo_ij->Cdbopi2;
          // fprintf(stderr,"\t dE/dbo: %f %f %f %f\n", bo_ij->Cdbos ,bo_ij->Cdbopi, bo_ij->Cdbopi2,bo_ij->Cdbo);
        } else {
          bo_ij->Cdbo += CEbo;
          bo_ij->Cdbopi -= (CEbo + twbp->De_p);
          bo_ij->Cdbopi2 -= (CEbo + twbp->De_pp);
        }
        /* Stabilisation terminal triple bond */
        if (bo_ij->BO >= 1.00) {
          if (gp37 == 2 ||
               (sbp_i->mass == 12.0000 && sbp_j->mass == 15.9990) ||
               (sbp_j->mass == 12.0000 && sbp_i->mass == 15.9990)) {
            exphu = exp(-gp7 * SQR(bo_ij->BO - 2.50));
            exphua1 = exp(-gp3 * (workspace->total_bond_order[i]-bo_ij->BO));
            exphub1 = exp(-gp3 * (workspace->total_bond_order[j]-bo_ij->BO));
            exphuov = exp(gp4 * (workspace->Delta[i] + workspace->Delta[j]));
            hulpov = 1.0 / (1.0 + 25.0 * exphuov);

            estriph = gp10 * exphu * hulpov * (exphua1 + exphub1);
            data->my_en.e_bond += estriph;

            decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1) *
              (gp3 - 2.0 * gp7 * (bo_ij->BO-2.50));
            decobdboua = -gp10 * exphu * hulpov *
              (gp3*exphua1 + 25.0*gp4*exphuov*hulpov*(exphua1+exphub1));
            decobdboub = -gp10 * exphu * hulpov *
              (gp3*exphub1 + 25.0*gp4*exphuov*hulpov*(exphua1+exphub1));

            /* tally energy into global or per-atom energy accumulators */
            if (system->pair_ptr->eflag_either)
              system->pair_ptr->ev_tally(i,j,natoms,1,estriph,0.0,0.0,0.0,0.0,0.0);

            bo_ij->Cdbo += decobdbo;
            workspace->CdDelta[i] += decobdboua;
            workspace->CdDelta[j] += decobdboub;
          }
        }
      }
    }
  }

}
