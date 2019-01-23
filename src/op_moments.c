#include "op_moments.h"

/**
 * Initialize the opQ struct
 * This is the operator that matches eddy currents
 */
void cvxop_moments_init(cvxop_moments *opQ, int N, int ind_inv, double dt,
                     double init_weight, int verbose)
{
    opQ->active = 1;
    opQ->N = N;
    opQ->ind_inv = ind_inv;
    opQ->dt = dt;
    opQ->verbose = verbose;

    opQ->Nrows = 0; // # of eddy current rows

    cvxmat_alloc(&opQ->Q0, MAXROWS, N);
    cvxmat_alloc(&opQ->Q, MAXROWS, N);

    cvxmat_alloc(&opQ->norms, MAXROWS, 1);
    cvxmat_alloc(&opQ->weights, MAXROWS, 1);
    cvxmat_alloc(&opQ->checks, MAXROWS, 1);
    cvxmat_alloc(&opQ->tolerances, MAXROWS, 1);
    cvxmat_alloc(&opQ->goals, MAXROWS, 1);
    cvxmat_alloc(&opQ->sigQ, MAXROWS, 1);

    cvxmat_alloc(&opQ->zQ, MAXROWS, 1);
    cvxmat_alloc(&opQ->zQbuff, MAXROWS, 1);
    cvxmat_alloc(&opQ->zQbar, MAXROWS, 1);
    cvxmat_alloc(&opQ->Qx, MAXROWS, 1);

    cvxmat_setvals(&opQ->weights, init_weight);
}


/**
 * Add a moment constraint
 * The *1000 x 3 are to get into units of mT/m*ms^X
 */
void cvxop_moments_addrow(cvxop_moments *opQ, int order, double goal, double tol)
{
    for (int i = 0; i < opQ->N; i++) {
        double ii = i;
        double val = 1000.0 * 1000.0 * opQ->dt * pow( (1000.0 * opQ->dt*ii), (double)order );
        if (i > opQ->ind_inv) {val = -val;}
        cvxmat_set(&(opQ->Q0), opQ->Nrows, i, val);
    }

    opQ->tolerances.vals[opQ->Nrows] = tol;
    opQ->goals.vals[opQ->Nrows] = goal;
    opQ->Nrows += 1;
}


/**
 * Scale Q to have unit norm rows, and calculate sigQ
 */
void cvxop_moments_finishinit(cvxop_moments *opQ)
{
    // Calculate the row norms of the eddy current array and store
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), j, i);
            opQ->norms.vals[j] += temp*temp;
        }
        opQ->norms.vals[j] = sqrt(opQ->norms.vals[j]);
    }

    // Scale row norms to 1.0
    for (int j = 0; j < opQ->Nrows; j++) {
        opQ->weights.vals[j] /= opQ->norms.vals[j];
    }

    // Make Q as weighted Q0
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), j, i);
            cvxmat_set(&(opQ->Q), j, i, opQ->weights.vals[j] * temp);
        }
    }


    // Calculate sigQ as inverse of sum(abs(row of Q))
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q), j, i);
            opQ->sigQ.vals[j] += fabs(temp);
        }
        opQ->sigQ.vals[j] = 1.0 / opQ->sigQ.vals[j];
    }
}


/*
 * Reweight the constraint and update all the subsequent weightings, and also the current descent direction zQ
 * basically weight_mod * Q
 */
void cvxop_moments_reweight(cvxop_moments *opQ, double weight_mod)
{
    double ww;
    for (int j = 0; j < opQ->Nrows; j++) {
        ww = 1.0;
        if (opQ->checks.vals[j] > 0) {
            ww = 2.0 * opQ->checks.vals[j];
            if (ww > weight_mod) {
                ww = weight_mod;
            }
        }
        opQ->weights.vals[j] *= ww;
        opQ->zQ.vals[j] *= ww;
    }

    // Make Q as weighted Q0
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), j, i);
            cvxmat_set(&(opQ->Q), j, i, opQ->weights.vals[j] * temp);
        }
    }

    // Calculate sigQ as inverse of sum(abs(row of Q))
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q), j, i);
            opQ->sigQ.vals[j] += fabs(temp);
        }
        opQ->sigQ.vals[j] = 1.0 / opQ->sigQ.vals[j];
    }
}


/**
 * Add absolute value of columns to the tau matrix 
 */
void cvxop_moments_add2tau(cvxop_moments *opQ, cvx_mat *tau_mat)
{
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q), j, i);
            tau_mat->vals[i] += fabs(temp);
        }
    }    
}



/**
 * Step the gradient waveform (taumx)
 */
void cvxop_moments_add2taumx(cvxop_moments *opQ, cvx_mat *taumx)
{   
    // MATH: taumx += E*zE
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q), j, i);
            taumx->vals[i] += (temp * opQ->zQ.vals[j]);
        }
    }

}


/**
 * Primal dual update
 */
void cvxop_moments_update(cvxop_moments *opQ, cvx_mat *txmx, double rr)
{
    if (opQ->Nrows > 0) {
        cvxmat_setvals(&(opQ->Qx), 0.0);

        // MATH: Ex = E * txmx
        for (int j = 0; j < opQ->Nrows; j++) {
            for (int i = 0; i < opQ->N; i++) {
                double temp = cvxmat_get(&(opQ->Q), j, i) * txmx->vals[i];
                opQ->Qx.vals[j] += temp;
            }
        }

        // MATH: Ex = Ex * sigE
        for (int j = 0; j < opQ->Nrows; j++) {
            opQ->Qx.vals[j] *= opQ->sigQ.vals[j];
        }

        // MATH: zEbuff  = zE + Ex = zE + sigE.*(E*txmx) 
        for (int j = 0; j < opQ->Nrows; j++) {
            opQ->zQbuff.vals[j] = opQ->zQ.vals[j] + opQ->Qx.vals[j];
        }

        // MATH: zEbar = clip( zEbuff/sigE , [-upper_tol, lower_tol])
        double cushion = 0.99;
        for (int j = 0; j < opQ->Nrows; j++) {
            double low =  (opQ->goals.vals[j] - cushion*opQ->tolerances.vals[j]) * opQ->weights.vals[j];
            double high = (opQ->goals.vals[j] + cushion*opQ->tolerances.vals[j]) * opQ->weights.vals[j];
            double val = opQ->zQbuff.vals[j] / opQ->sigQ.vals[j];
            if (val < low) {
                opQ->zQbar.vals[j] = low;
            } else if (val > high) {
                opQ->zQbar.vals[j] = high;
            } else {
                opQ->zQbar.vals[j] = val;
            }
            
        }

        // MATH: zEbar = zEbuff - sigE*zEbar
        for (int j = 0; j < opQ->Nrows; j++) {
            opQ->zQbar.vals[j] = opQ->zQbuff.vals[j] - opQ->sigQ.vals[j] * opQ->zQbar.vals[j];
        }

        // MATH: zE = rr*zEbar + (1-rr)*zE
        for (int j = 0; j < opQ->Nrows; j++) {
            opQ->zQ.vals[j] = rr * opQ->zQbar.vals[j] + (1 - rr) * opQ->zQ.vals[j];
        }
    }
}




/*
 * Check if moments are larger than a fixed tolerance
 */
int cvxop_moments_check(cvxop_moments *opQ, cvx_mat *G)
{
    cvxmat_setvals(&(opQ->Qx), 0.0);

    // MATH: Ex = E * txmx
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), j, i) * G->vals[i];
            opQ->Qx.vals[j] += temp;
        }
    }

    // Set checks to be 0 if within tolerance, otherwise set to the ratio of eddy current to tolerance
    cvxmat_setvals(&(opQ->checks), 0.0);
    for (int j = 0; j < opQ->Nrows; j++) {
        double tol = opQ->tolerances.vals[j];
        double low =  opQ->goals.vals[j] - tol;
        double high = opQ->goals.vals[j] + tol;
        double diff;
        if (opQ->Qx.vals[j] < low) {
            diff = opQ->goals.vals[j] - opQ->Qx.vals[j];
            opQ->checks.vals[j] = diff / tol;
        } else if (opQ->Qx.vals[j] > high) {
            diff = opQ->Qx.vals[j] - opQ->goals.vals[j];
            opQ->checks.vals[j] = diff / tol;
        }
    }


    int moments_bad = 0;

    for (int j = 0; j < opQ->Nrows; j++) {
        if (opQ->checks.vals[j] > 0) {
             moments_bad = 1;
        }
    }

    if (opQ->verbose>0) {   
        printf("    Moments check:  (%d)  %.2e  %.2e  %.2e \n", moments_bad, opQ->Qx.vals[0], opQ->Qx.vals[1], opQ->Qx.vals[2]);
    }

    return moments_bad;
}


/*
 * Free memory
 */
void cvxop_moments_destroy(cvxop_moments *opQ)
{

    free(opQ->norms.vals);
    free(opQ->weights.vals);
    free(opQ->checks.vals);
    free(opQ->tolerances.vals);
    free(opQ->goals.vals);


    free(opQ->Q0.vals);
    free(opQ->Q.vals);

    free(opQ->sigQ.vals);

    free(opQ->zQ.vals);
    free(opQ->zQbuff.vals);
    free(opQ->zQbar.vals);
    free(opQ->Qx.vals);
}
