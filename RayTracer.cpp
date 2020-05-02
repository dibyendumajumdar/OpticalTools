#include <memory>
#include <string>
#include <vector>

#include <cassert>
#include <cstdio>
#include <cmath>

struct SkewRayTransferInput {
    double X_1; /* x coordinate on origin surface */
    double Y_1; /* y coordinate on origin surface */
    double Z_1; /* z coordinate on origin surface */
    double K_1; /* product of nj and direction cosine */
    double L_1; /* product of nj and direction cosine */
    double M_1; /* product of nj and direction cosine */
    double t_1;
    double n_1;
    double n;
    double c; /* curvature of surface */
    double e,f,g,h; /* aspheric coefficients */
};

struct SkewRayTranserOutput {
    double X;
    double Y;
    double Z;
    double K;
    double L;
    double M;
};

static inline double square(double x) { return x*x; }

/*
 * The code below is a direct implementation of section 5.4.6
 * 'Summary of ray trace equations', in chapter 5. Fundamental
 * Method of Ray Tracing, in MIL-HDBK-141. Test cases are taken from
 * the worked examples.
 */
void calculate_skew_ray_transfer(SkewRayTransferInput *in, SkewRayTranserOutput *out) {
    double d_1_over_n_1 = (in->t_1 - in->Z_1) / in->M_1; //  (1)
    double Y_T = in->Y_1 + d_1_over_n_1 * in->L_1; // (2)
    double X_T = in->X_1 + d_1_over_n_1 * in->K_1; // (3)
    double H = in->c * (square(X_T) + square(Y_T)); // (9)
    double B = in->M_1 - in->c * (Y_T * in->L_1 + X_T * in->K_1); // (10)
    double n_1_cos_I = in->n_1 * std::sqrt( square(B / in->n_1) - in->c * H ); // (7)
    double A_over_n_1 = H / (B + n_1_cos_I); // (8)
    out->X = X_T + A_over_n_1 * in->K_1; // (4)
    out->Y = Y_T + A_over_n_1 * in->L_1; // (5)
    out->Z = A_over_n_1 * in->M_1; // (6)

    double n_cos_Idash = in->n * std::sqrt(square(n_1_cos_I/in->n) - square(in->n_1/in->n) + 1); // (11)
    double Tao = n_cos_Idash - n_1_cos_I; // (12)
    out->K = in->K_1 - out->X * in->c * Tao; // (13)
    out->L = in->L_1 - out->Y * in->c * Tao; // (14)
    out->M = in->M_1 - ( out->Z * in->c - 1) * Tao; // (15)
}

void test_skew_transfer1() {
    SkewRayTransferInput in;
    SkewRayTranserOutput out;
    in.c = 0.25284872;
    in.t_1 = -2.2;
    in.n_1 = 1.0;
    in.X_1 = 1.48;
    in.Y_1 = 0.0;
    in.Z_1 = 0.0;
    in.K_1 = 0.0;
    in.L_1 = 0.1736;
    in.M_1 = 0.98481625;
    in.n = 1.62;
    calculate_skew_ray_transfer(&in, &out);
}

/*
 * The code below is a direct implementation of section 5.5.6
 * 'Summary of ray trace equations', in chapter 5. Fundamental
 * Method of Ray Tracing, in MIL-HDBK-141. Test cases are taken from
 * the worked examples.
 */
void calculate_skew_ray_transfer_aspheric(SkewRayTransferInput *in, SkewRayTranserOutput *out) {
    double d_1_over_n_1 = (in->t_1 - in->Z_1) / in->M_1; //  (1)
    double Y_T = in->Y_1 + d_1_over_n_1 * in->L_1; // (2)
    double X_T = in->X_1 + d_1_over_n_1 * in->K_1; // (3)
    double H = in->c * (square(X_T) + square(Y_T)); // (9)
    double B = in->M_1 - in->c * (Y_T * in->L_1 + X_T * in->K_1); // (10)
    double n_1_cos_I = in->n_1 * std::sqrt( square(B / in->n_1) - in->c * H ); // (7)
    double A_over_n_1 = H / (B + n_1_cos_I); // (8)
    double Xn = X_T + A_over_n_1 * in->K_1; // (4)
    double Yn = Y_T + A_over_n_1 * in->L_1; // (5)
    double Zn = A_over_n_1 * in->M_1; // (6)
    double U,V,W;
    // iterate
    for (int i = 0; i < 100; i++) {
        // S^2 = X^2 + Y^2
        double Sn_sq = square(Xn) + square(Yn);// (16)
        double c_sq_Sn_sq = square(in->c) * Sn_sq;
        // W = (1 - c^2 * S^2)^(1/2)
        W = std::sqrt(1 - c_sq_Sn_sq);// (18)
        double c_over_1_plus_W = in->c / (1 + W);
        double hS2 = Sn_sq*in->h;
        double hS4 = hS2 * Sn_sq;
        double hS6 = hS4 * Sn_sq;
        double hS8 = hS6 * Sn_sq;
        double hS10 = hS8 * Sn_sq;
        double gS2 = Sn_sq * in->g;
        double gS4 = gS2 * Sn_sq;
        double gS6 = gS4 * Sn_sq;
        double gS8 = gS6 * Sn_sq;
        double fS2 = Sn_sq * in->f;
        double fS4 = fS2 * Sn_sq;
        double fS6 = fS4 * Sn_sq;
        double eS2 = Sn_sq * in->e;
        double eS4 = eS2 * Sn_sq;
        double F = Zn - (hS10 + gS8 + fS6 + eS4 + Sn_sq*c_over_1_plus_W);
        // Fn is different in Handbook
        //double F = Zn - ((in->c * Sn_sq / (1 + c_sq_Sn_sq)) + in->e * S4 + in->f * S6 + in->g * S8 + in->h * S10);// (17)
        double minusF = -F;
        double E_plus = 10*hS8 + 8*gS6 + 6*fS4 + 4*eS2;
        double E = in->c + W * E_plus;              // (22)
        U = -Xn * E;                                                                                       // (23)
        V = -Yn * E;                                                                                       // (24)
        double deltaA_over_n_1 = (-F * W) / (in->K_1 * U + in->L_1 * V + in->M_1 * W);                     // (25)
        Xn = Xn + deltaA_over_n_1 * in->K_1;                                                               // (19)
        Yn = Yn + deltaA_over_n_1 * in->L_1;                                                               // (20)
        Zn = Zn + deltaA_over_n_1 * in->M_1;                                                               // (21)
        if (std::fabs(deltaA_over_n_1) < 1e-10) {
            break;
        }
    }
    out->X = Xn;
    out->Y = Yn;
    out->Z = Zn;
    double G2 = square(U) + square(V) + square(W); // (26)
    double G_n_1_cos_I = in->K_1 * U + in->L_1 * V + in->M_1 * W; // (27)
    double Gn_cos_Idash = in->n * std::sqrt(square(G_n_1_cos_I/in->n) - G2*square(in->n_1/in->n) + G2); // (28)
    double P = ( Gn_cos_Idash - G_n_1_cos_I) / G2; // (29)
    out->K = in->K_1 + U*P; // (30)
    out->L = in->L_1 + V*P; // (31)
    out->M = in->M_1 + W*P; // (32)
    fprintf(stdout, "%.16g %.16g %.16g\n", out->K, out->L, out->M);
}

void test_skew_transfer1_aspheric() {
    SkewRayTransferInput in;
    SkewRayTranserOutput out;
    in.c = 0.25284872;
    in.t_1 = -2.2;
    in.n_1 = 1.0;
    in.X_1 = 1.48;
    in.Y_1 = 0.0;
    in.Z_1 = 0.0;
    in.K_1 = 0.0;
    in.L_1 = 0.1736;
    in.M_1 = 0.98481625;
    in.n = 1.62;
    in.e = -0.005;
    in.f = 0.00001;
    in.g = -0.0000005;
    in.h = 0;
    calculate_skew_ray_transfer_aspheric(&in, &out);
    fprintf(stdout, "Expected -0.2048155926643451 0.2205208021083463 1.591798088673029\n");
}


int main(int argc, const char *argv[]) {
    test_skew_transfer1();
    test_skew_transfer1_aspheric();
    return 0;
}

