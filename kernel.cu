#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include "Header.h"

#define Omega 0.0
#define N 1536 // 7167 //1792 //1792                 // ���������� ����� �� x
#define M 1536  // //1280 //1280                 // ���������� ����� �� y
#define K (N*M)                // ���������� ����� � �����
#define x_max 5.0 //450.0
#define x_min (x_max/(2.0 * N)) // -2760.0 // -2500.0 // -1300  //-2000                // -1500.0
#define y_max 5.0 // 2250.0 // 1600.0 //1840.0
#define y_min (y_max/(2.0 * M))  // -30.0 // (y_max/(2.0 * M)) 
#define dx ((x_max)/(N))  // ((x_max - x_min)/(N - 1))     // �������� ����� �� dx
#define dy ((y_max)/(M)) //  ((y_max - y_min)/(M - 1))     // �������� ����� �� dy

#define ER_S std::cout << "\n---------------------\nStandart error in file: Solvers.cpp\n" << endl
#define watch(x) cout << (#x) << " is " << (x) << endl
#define eps (1e-10)
#define eps8 (1e-8)
#define hy 00.0
#define hx -3288.0
#define grad_p true
#define Nmin 4              // ������ ����� ����� �������?
#define THREADS_PER_BLOCK 256    // ���������� ����� � ����� ������ // ����������, ����� ���������� ����� � ����� �������� �� ����� ����� (����� N �������� �� ����� �����)

__device__ int sign(double& x);
__device__ double minmod(double x, double y);
__device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y);
__device__ void linear2(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B);
__device__ void lev(const double& enI, const double& pI, const double& rI, const double& enII,//
    const double& pII, const double& rII, double& uuu, double& fee);
__device__ void devtwo(const double& enI, const double& pI, const double& rI, const double& enII, const double& pII, const double& rII, //
    const double& w, double& p);
__device__ void newton(const double& enI, const double& pI, const double& rI, const double& enII, const double& pII, const double& rII, //
    const double& w, double& p);
__device__ void perpendicular(double a1, double a2, double a3, double& b1, double& b2, double& b3, //
    double& c1, double& c2, double& c3, bool t);
__device__ double Godunov_Solver_Alexashov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad);
__host__ bool areaa(double x, double y, double ro, double p, double u, double v);

using namespace std;

__device__ double minmod(double x, double y)
{
    if (sign(x) + sign(y) == 0)
    {
        return 0.0;
    }
    else
    {
        return   ((sign(x) + sign(y)) / 2.0) * min(fabs(x), fabs(y));  ///minmod
        //return (2*x*y)/(x + y);   /// vanleer
    }
}

__device__ double VanLier(double x, double y)
{
    return minmod((x + y)/2.0, 2.0 * minmod(x, y));
}

__device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y)
{
    double d = minmod((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    return  (d * (y - x2) + t2);
}

__device__ void linear2(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B)
{
    // ������� �������� - ����������� - �� ������� �� ����
    double d = minmod((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    A = (d * (y1 - x2) + t2);
    B = (d * (y2 - x2) + t2);
    //printf("%lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf \n", x1, t1, x2, t2, x3, t3, y1, y2, A, B);
    return;
}

__device__ int sign(double& x)
{
    if (x > 0)
    {
        return 1;
    }
    else if (x < 0)
    {
        return  -1;
    }
    else
    {
        return 0;
    }
}

__device__ double  my_min(double a, double b)
{
    if (a <= b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

__device__ double  my_max(double a, double b)
{
    if (a >= b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

double polar_angle(double x, double y)
{
    if (x * x + y * y < 0.0000001)
    {
        return 0.0;
    }
    else if (x < 0)
    {
        return atan(y / x) + 1.0 * PI;
    }
    else if (x > 0 && y >= 0)
    {
        return atan(y / x);
    }
    else if (x > 0 && y < 0)
    {
        return atan(y / x) + 2.0 * PI;
    }
    else if (y > 0 && x >= 0 && x <= 0)
    {
        return PI / 2.0;
    }
    else if (y < 0 && x >= 0 && x <= 0)
    {
        return  3.0 * PI / 2.0;
    }
    return 0.0;
}

void dekard_skorost(double x, double y, double z, double Vr, double Vphi, double Vtheta, double& Vx, double& Vy, double& Vz)
{
    double r_2 = sqrt(x * x + y * y + z * z);
    double the_2 = acos(z / r_2);
    double phi_2 = polar_angle(x, y);

    if (sqrt(x * x + y * y) < 0.000001)
    {
        Vx = 0.0;
        Vy = 0.0;
        Vz = 0.0;
    }
    else
    {
        Vx = Vr * sin(the_2) * cos(phi_2) + Vtheta * cos(the_2) * cos(phi_2) - Vphi * sin(phi_2);
        Vy = Vr * sin(the_2) * sin(phi_2) + Vtheta * cos(the_2) * sin(phi_2) + Vphi * cos(phi_2);
        Vz = Vr * cos(the_2) - Vtheta * sin(the_2);
    }
}

__host__ bool areaa(double  x, double y, double ro, double p, double u, double v)
{
    if (ro <= 0.0)
    {
        return true;
    }
    double Max = sqrt((u * u + v * v) / (ggg * p / ro));
    double T = p / ro;
    if ((x < 36.8) && (y < 336))
    {
        return true;
    }
    if (( fabs(ro - 1.0) < 0.000001) && (fabs(Max - 3.0) < 0.000001))
    {
        return false;
    }
    if ((x > 240.3)||(y > 616.4) )
    {
        return false;
    }
    if ((x < -368) && ( T > 0.12))
    {
        return true;
    }
    if (M > 3.3)
    {
        return true;
    }
    if ((x > 1.0)&&(ro < 1.7))
    {
        return true;
    }
    else
    {
        return false;
    }
    return false;
}

__device__ double Godunov_Solver_Alexashov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double w = 0.0;
    double al = n1;
    double be = n2;
    double ge = 0.0;
    double time = 0.0;

    double al2 = -n2;
    double be2 = n1;
    double ge2 = 0.0;
    double al3 = 0.0;
    double be3 = 0.0;
    double ge3 = 1.0;

    double enI = al * Lu.x + be * Lu.y;
    double teI2 = al2 * Lu.x + be2 * Lu.y;
    double teI3 = al3 * Lu.x + be3 * Lu.y;
    double enII = al * Ru.x + be * Ru.y;
    double teII2 = al2 * Ru.x + be2 * Ru.y;
    double teII3 = al3 * Ru.x + be3 * Ru.y;

    double pI = Ls.y;
    double pII = Rs.y;
    double rI = Ls.x;
    double rII = Rs.x;

    int ipiz = 0;
    if (pI > pII)   // ����� ������� �������
    {
        double eno2 = enII;;
        double teo22 = teII2;
        double teo23 = teII3;
        double p2 = pII;
        double r2 = rII;

        double eno1 = enI;
        double teo12 = teI2;
        double teo13 = teI3;
        double p1 = pI;
        double r1 = rI;

        enI = -eno2;
        teI2 = teo22;
        teI3 = teo23;
        pI = p2;
        rI = r2;

        enII = -eno1;
        teII2 = teo12;
        teII3 = teo13;
        pII = p1;
        rII = r1;
        w = -w;
        ipiz = 1;                                                                // ???? �� ����� ����� ������ ����?
    }

    double cI = 0.0;
    double cII = 0.0;
    if (rI != 0.0)
    {
        cI = __dsqrt_rn(ga * pI / rI);
    }
    if (rII != 0.0)
    {
        cII = __dsqrt_rn(ga * pII / rII);
    }

   /* printf("C2 !!!! = %lf =  kor  %lf \n", cII, ga * pII / rII);
    printf("%lf , %lf, %lf \n",ga,pII,rII);*/

    double a = __dsqrt_rn(rI * (g2 * pII + g1 * pI) / 2.0);
    double Uud = (pII - pI) / a;
    double Urz = -2.0 * cII / g1 * (1.0 - pow((pI / pII), gm));
    double Uvk = -2.0 * (cII + cI) / g1;
    double Udf = enI - enII;

    int il, ip;
    double p, r, te2, te3, en;

    if (Udf < Uvk)
    {
        il = -1;
        ip = -1;
    }
    else if ((Udf >= Uvk) && (Udf <= Urz))
    {
        p = pI * pow(((Udf - Uvk) / (Urz - Uvk)), (1.0 / gm));
        il = 0;
        ip = 0;
    }
    else if ((Udf > Urz) && (Udf <= Uud))
    {
        devtwo(enI, pI, rI, enII, pII, rII, w, p);
        il = 1;
        ip = 0;
    }
    else if (Udf > Uud)
    {
        newton(enI, pI, rI, enII, pII, rII, w, p);
        il = 1;
        ip = 1;
    }

    //*********TWO SHOCKS**********************************************
    if ((il == 1) && (ip == 1))
    {
       /* printf("TWO SHOCKS\n");*/
        double aI = __dsqrt_rn(rI * (g2 / 2.0 * p + g1 / 2.0 * pI));
        double aII = __dsqrt_rn(rII * (g2 / 2.0 * p + g1 / 2.0 * pII));

        double u = (aI * enI + aII * enII + pI - pII) / (aI + aII);
        double dI = enI - aI / rI;
        double dII = enII + aII / rII;


        double UU = max(fabs(dI), fabs(dII));
        if (UU > eps8)
        {
            time = krit * rad / UU;
        }
        else
        {
            time = krit * rad / eps8;
        }


        if (w <= dI)
        {
            en = enI;
            p = pI;
            r = rI;
            te2 = teI2;
            te3 = teI3;
        }
        else if ((w > dI) && (w <= u))
        {
            en = u;
            p = p;
            r = rI * aI / (aI - rI * (enI - u));
            te2 = teI2;
            te3 = teI3;
        }
        else if ((w > u) && (w < dII))
        {
            en = u;
            p = p;
            r = rII * aII / (aII + rII * (enII - u));
            te2 = teII2;
            te3 = teII3;
        }
        else if (w >= dII)
        {
            en = enII;
            p = pII;
            r = rII;
            te2 = teII2;
            te3 = teII3;
        }
    }


    //*********LEFT - SHOCK, RIGHT - EXPANSION FAN*******************
    if ((il == 1) && (ip == 0))
    {
        //printf("LEFT - SHOCK, RIGHT - EXPANSION FAN\n");
        double aI = __dsqrt_rn(rI * (g2 / 2.0 * p + g1 / 2.0 * pI));
        double aII;
        if (fabs(p - pII) < eps)
        {
            aII = rII * cII;
        }
        else
        {
            aII = gm * rII * cII * (1.0 - p / pII) / (1.0 - pow((p / pII), gm));
        }

        double u = (aI * enI + aII * enII + pI - pII) / (aI + aII);
        double dI = enI - aI / rI;
        double dII = enII + cII;
        double ddII = u + cII - g1 * (enII - u) / 2.0;

        double UU = max(fabs(dI), fabs(dII));
        UU = max(UU, fabs(ddII));
        if (UU > eps8)
        {
            time = krit * rad / UU;
        }
        else
        {
            time = krit * rad / eps8;
        }

        if (w <= dI)
        {
            en = enI;
            p = pI;
            r = rI;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > dI) && (w <= u))
        {
            en = u;
            p = p;
            r = rI * aI / (aI - rI * (enI - u));
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > u) && (w <= ddII))
        {
            double ce = cII - g1 / 2.0 * (enII - u);
            en = u;
            p = p;
            r = ga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if ((w > ddII) && (w < dII))
        {
            double ce = -g1 / g2 * (enII - w) + 2.0 / g2 * cII;
            en = w - ce;
            p = pII * pow((ce / cII), (1.0 / gm));
            r = ga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if (w >= dII)
        {
            en = enII;
            p = pII;
            r = rII;
            te2 = teII2;
            te3 = teII3;
        }
    }
    //*********TWO EXPANSION FANS**************************************
    if ((il == 0) && (ip == 0))
    {
        //printf("TWO EXPANSION FANS\n");
        double aI;
        //printf("p = %lf\n", p);
        if (fabs(p - pI) < eps)
        {
            aI = rI * cI;
        }
        else
        {
            aI = gm * rI * cI * (1.0 - p / pI) / (1.0 - pow((p / pI), gm));
        }
        //printf("aI = %lf\n", aI);

        double aII;
        if (fabs(p - pII) < eps)
        {
            aII = rII * cII;
        }
        else
        {
            aII = gm * rII * cII * (1.0 - p / pII) / (1.0 - pow((p / pII), gm));
        }

        //printf("aII = %lf\n", aI);

        double u = (aI * enI + aII * enII + pI - pII) / (aI + aII);
        double dI = enI - cI;
        double ddI = u - cI - g1 * (enI - u) / 2.0;
        double dII = enII + cII;
        double ddII = u + cII - g1 * (enII - u) / 2.0;
        /*printf("enII = %lf\n", enII);
        printf("cII = %lf\n", cII);
        printf("u = %lf\n", u);
        printf("dI = %lf\n", dI);
        printf("dII = %lf\n", dII);
        printf("ddI = %lf\n", ddI);
        printf("ddII = %lf\n", ddII);*/

        double UU = max(fabs(dI), fabs(dII));
        UU = max(UU, fabs(ddII));
        UU = max(UU, fabs(ddI));
        if (UU > eps8)
        {
            time = krit * rad / UU;
        }
        else
        {
            time = krit * rad / eps8;
        }


        if (w <= dI)
        {
            //printf("1\n");
            en = enI;
            p = pI;
            r = rI;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > dI) && (w < ddI))
        {
            //printf("2\n");
            double ce = g1 / g2 * (enI - w) + 2.0 / g2 * cI;
            en = w + ce;
            p = pI * pow((ce / cI), (1.0 / gm));
            r = ga * p / ce / ce;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w >= ddI) && (w <= u))
        {
            //printf("3\n");
            double ce = cI + g1 / 2.0 * (enI - u);
            en = u;
            p = p;
            r = ga * p / ce / ce;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > u) && (w <= ddII))
        {
            //printf("4\n");
            double ce = cII - g1 / 2.0 * (enII - u);
            en = u;
            p = p;
            r = ga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if ((w > ddII) && (w < dII))
        {
            //printf("5\n");
            double ce = -g1 / g2 * (enII - w) + 2.0 / g2 * cII;
            en = w - ce;
            p = pII * pow((ce / cII), (1.0 / gm));
            r = ga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if (w >= dII)
        {
            //printf("6\n");
            en = enII;
            p = pII;
            r = rII;
            te2 = teII2;
            te3 = teII3;
        }
    }

    //*********VAKUUM ************************************************
    if ((il == -1) && (ip == -1))
    {
        //printf("VAKUUM\n");
        double dI = enI - cI;
        double ddI = enI + 2.0 / gg1 * cI;
        double dII = enII + cII;
        double ddII = enII - 2.0 / gg1 * cII;


        double UU = max(fabs(dI), fabs(dII));
        UU = max(UU, fabs(ddII));
        UU = max(UU, fabs(ddI));
        if (UU > eps8)
        {
            time = krit * rad / UU;
        }
        else
        {
            time = krit * rad / eps8;
        }


        if (w <= dI)
        {
            en = enI;
            p = pI;
            r = rI;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > dI) && (w < ddI))
        {
            double ce = gg1 / gg2 * (enI - w) + 2.0 / gg2 * cI;
            en = w + ce;
            p = pI * pow((ce / cI), (1.0 / gm));
            r = gga * p / ce / ce;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w >= ddI) && (w <= ddII))
        {
            en = w;
            p = 0.0;
            r = 0.0;
            te2 = 0.0;
            te3 = 0.0;
        }
        if ((w > ddII) && (w < dII))
        {
            double ce = -gg1 / gg2 * (enII - w) + 2.0 / gg2 * cII;
            en = w - ce;
            p = pII * pow((ce / cII), (1.0 / gm));
            r = gga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if (w >= dII)
        {
            en = enII;
            p = pII;
            r = rII;
            te2 = teII2;
            te3 = teII3;
        }
    }


    if (ipiz == 1)
    {
        en = -en;
        w = -w;
    }

    double uo = al * en + al2 * te2 + al3 * te3;
    double vo = be * en + be2 * te2 + be3 * te3;
    double wo = ge * en + ge2 * te2 + ge3 * te3;


    double eo = p / g1 + 0.5 * r * (uo * uo + vo * vo + wo * wo);
    en = al * uo + be * vo + ge * wo;

    Ps.x = (r * (en - w));
    Pu.x = (r * (en - w) * uo + al * p);
    Pu.y = (r * (en - w) * vo + be * p);
    //qqq[3] = (r * (en - w) * wo + ge * p);
    Ps.y = ((en - w) * eo + en * p);


    return time;

}

__device__ void perpendicular(double a1, double a2, double a3, double& b1, double& b2, double& b3, //
    double& c1, double& c2, double& c3, bool t)
{
    if (t == false)
    {
        double A = a1 * a1 + a2 * a2;
        if (A > 0.01 * (A + a3 * a3))
        {
            double B = sqrt(A);
            b1 = -a2 / B;
            b2 = a1 / B;
            b3 = 0.0;
            double C = sqrt(A * (A + a3 * a3));
            c1 = -a1 * a3 / C;
            c2 = -a2 * a3 / C;
            c3 = A / C;
            return;
        }
        A = a1 * a1 + a3 * a3;
        if (A > 0.01 * (A + a2 * a2))
        {
            double B = sqrt(A);
            b1 = -a3 / B;
            b2 = 0.0;
            b3 = a1 / B;
            double C = sqrt(A * (A + a2 * a2));
            c1 = a1 * a2 / C;
            c2 = -A / C;
            c3 = a2 * a3 / C;
            return;
        }
    }
    else
    {
        double A = a1 * a1 + a2 * a2;
        if (A > 0.01)
        {
            double B = sqrt(A);
            b1 = -a2 / B;
            b2 = a1 / B;
            b3 = 0.0;;
            c1 = -a1 * a3 / B;
            c2 = -a2 * a3 / B;
            c3 = A / B;
            return;
        }
        A = a1 * a1 + a3 * a3;
        if (A > 0.01)
        {
            double B = sqrt(A);
            b1 = -a3 / B;
            b2 = 0.0;
            b3 = a1 / B;

            c1 = a1 * a2 / B;
            c2 = -A / B;
            c3 = a2 * a3 / B;
            return;
        }
    }

}

__device__ void newton(const double& enI, const double& pI, const double& rI, const double& enII, const double& pII, const double& rII, //
    const double& w, double& p)
{
    double fI, fIs, fII, fIIs;
    double cI = __dsqrt_rn(ga * pI / rI);
    double cII = __dsqrt_rn(ga * pII / rII);
    double pn = pI * rII * cII + pII * rI * cI + (enI - enII) * rI * cI * rII * cII;
    pn = pn / (rI * cI + rII * cII);

    double pee = pn;

    int kiter = 0;
a1:
    p = pn;
    if (p <= 0.0)
    {
        printf("84645361\n");
    }

    kiter = kiter + 1;

    fI = (p - pI) / (rI * cI * __dsqrt_rn(gp * p / pI + gm));
    fIs = (ga + 1.0) * p / pI + (3.0 * ga - 1.0);
    fIs = fIs / (4.0 * ga * rI * cI * pow((gp * p / pI + gm), (3.0 / 2.0)));

    fII = (p - pII) / (rII * cII * __dsqrt_rn(gp * p / pII + gm));
    fIIs = (ga + 1.0) * p / pII + (3.0 * ga - 1.0);
    fIIs = fIIs / (4.0 * ga * rII * cII * pow((gp * p / pII + gm), (3.0 / 2.0)));


    if (kiter == 1100)
    {
        printf("0137592\n");
    }

    pn = p - (fI + fII - (enI - enII)) / (fIs + fIIs);

    if (fabs(pn / pee - p / pee) >= eps)
    {
        goto a1;
    }

    p = pn;

    return;
}

__device__ void devtwo(const double& enI, const double& pI, const double& rI, const double& enII, const double& pII, const double& rII, //
    const double& w, double& p)
{
    const double epsil = 10e-10;
    double kl, kp, kc, ksi, ksir, um, ksit;
    int kpizd;

    kl = pI;
    kp = pII;


    lev(enI, pI, rI, enII, pII, rII, kl, ksi);
    lev(enI, pI, rI, enII, pII, rII, kp, ksir);

    if (fabs(ksi) <= epsil)
    {
        um = kl;
        goto a1;
    }

    if (fabs(ksir) <= epsil)
    {
        um = kp;
        goto a1;
    }

    kpizd = 0;

a2:
    kpizd = kpizd + 1;

    if (kpizd == 1100)
    {
        printf("121421414\n");
        printf("%lf, %lf,%lf,%lf,%lf,%lf,\n", enI, pI, rI, enII, pII, rII);
    }


    kc = (kl + kp) / 2.0;

    lev(enI, pI, rI, enII, pII, rII, kc, ksit);

    if (fabs(ksit) <= epsil)
    {
        goto a3;
    }

    if ((ksi * ksit) <= 0.0)
    {
        kp = kc;
        ksir = ksit;
    }
    else
    {
        kl = kc;
        ksi = ksit;
    }

    goto a2;

a3:
    um = kc;
a1:

    p = um;

    return;
}

__device__ void lev(const double& enI, const double& pI, const double& rI, const double& enII,//
    const double& pII, const double& rII, double& uuu, double& fee)
{
    double cI = __dsqrt_rn(ga * pI / rI);
    double cII = __dsqrt_rn(ga * pII / rII);

    double fI = (uuu - pI) / (rI * cI * __dsqrt_rn(gp * uuu / pI + gm));

    double fII = 2.0 / g1 * cII * (pow((uuu / pII), gm) - 1.0);

    double f1 = fI + fII;
    double f2 = enI - enII;
    fee = f1 - f2;
    return;
}

__device__ double HLLC_Aleksashov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double n[3];
    n[0] = n1;
    n[1] = n2;
    n[2] = 0.0;
    //int id_bn = 1;
    //int n_state = 1;
    double FR[8], FL[8];
    double UL[8], UZ[8], UR[8];
    double UZL[8], UZR[8];

    double vL[3], vR[3], bL[3], bR[3];
    double vzL[3], vzR[3], bzL[3], bzR[3];
    double qv[3];
    double aco[3][3];

    double wv = 0.0;
    double r1 = Ls.x;
    double u1 = Lu.x;
    double v1 = Lu.y;
    double w1 = 0.0;
    double p1 = Ls.y;
    double bx1 = 0.0;
    double by1 = 0.0;
    double bz1 = 0.0;


    double r2 = Rs.x;
    double u2 = Ru.x;
    double v2 = Ru.y;
    double w2 = 0.0;
    double p2 = Rs.y;
    double bx2 = 0.0;
    double by2 = 0.0;
    double bz2 = 0.0;

    double ro = (r2 + r1) / 2.0;
    double ap = (p2 + p1) / 2.0;
    double abx = (bx2 + bx1) / 2.0;
    double aby = (by2 + by1) / 2.0;
    double abz = (bz2 + bz1) / 2.0;


    double bk = abx * n[0] + aby * n[1] + abz * n[2];
    double b2 = kv(abx) + kv(aby) + kv(abz);

    double d = b2 - kv(bk);
    aco[0][0] = n[0];
    aco[1][0] = n[1];
    aco[2][0] = n[2];
    if (d > eps)
    {
        d = __dsqrt_rn(d);
        aco[0][1] = (abx - bk * n[0]) / d;
        aco[1][1] = (aby - bk * n[1]) / d;
        aco[2][1] = (abz - bk * n[2]) / d;
        aco[0][2] = (aby * n[2] - abz * n[1]) / d;
        aco[1][2] = (abz * n[0] - abx * n[2]) / d;
        aco[2][2] = (abx * n[1] - aby * n[0]) / d;
    }
    else
    {
        double aix, aiy, aiz;
        if ((fabs(n[0]) < fabs(n[1])) && (fabs(n[0]) < fabs(n[2])))
        {
            aix = 1.0;
            aiy = 0.0;
            aiz = 0.0;
        }
        else if (fabs(n[1]) < fabs(n[2]))
        {
            aix = 0.0;
            aiy = 1.0;
            aiz = 0.0;
        }
        else
        {
            aix = 0.0;
            aiy = 0.0;
            aiz = 1.0;
        }

        double aik = aix * n[0] + aiy * n[1] + aiz * n[2];
        d = __dsqrt_rn(1.0 - kv(aik));
        aco[0][1] = (aix - aik * n[0]) / d;
        aco[1][1] = (aiy - aik * n[1]) / d;
        aco[2][1] = (aiz - aik * n[2]) / d;
        aco[0][2] = (aiy * n[2] - aiz * n[1]) / d;
        aco[1][2] = (aiz * n[0] - aix * n[2]) / d;
        aco[2][2] = (aix * n[1] - aiy * n[0]) / d;
    }

    for (int i = 0; i < 3; i++)
    {
        vL[i] = aco[0][i] * u1 + aco[1][i] * v1 + aco[2][i] * w1;
        vR[i] = aco[0][i] * u2 + aco[1][i] * v2 + aco[2][i] * w2;
        bL[i] = aco[0][i] * bx1 + aco[1][i] * by1 + aco[2][i] * bz1;
        bR[i] = aco[0][i] * bx2 + aco[1][i] * by2 + aco[2][i] * bz2;
    }

    double aaL = bL[0] / __dsqrt_rn(r1);
    double b2L = kv(bL[0]) + kv(bL[1]) + kv(bL[2]);
    double b21 = b2L / r1;
    double cL = __dsqrt_rn(ga * p1 / r1);
    double qp = __dsqrt_rn(b21 + cL * (cL + 2.0 * aaL));
    double qm = __dsqrt_rn(b21 + cL * (cL - 2.0 * aaL));
    double cfL = (qp + qm) / 2.0;
    double ptL = p1 + b2L / 2.0;

    double aaR = bR[0] / __dsqrt_rn(r2);
    double b2R = kv(bR[0]) + kv(bR[1]) + kv(bR[2]);
    double b22 = b2R / r2;
    double cR = __dsqrt_rn(ga * p2 / r2);
    qp = __dsqrt_rn(b22 + cR * (cR + 2.0 * aaR));
    qm = __dsqrt_rn(b22 + cR * (cR - 2.0 * aaR));
    double cfR = (qp + qm) / 2.0;
    double ptR = p2 + b2R / 2.0;

    double aC = (aaL + aaR) / 2.0;
    double b2o = (b22 + b21) / 2.0;
    double cC = __dsqrt_rn(ga * ap / ro);
    qp = __dsqrt_rn(b2o + cC * (cC + 2.0 * aC));
    qm = __dsqrt_rn(b2o + cC * (cC - 2.0 * aC));
    double cfC = (qp + qm) / 2.0;
    double vC1 = (vL[0] + vR[0]) / 2.0;

    double SL = min((vL[0] - cfL), (vR[0] - cfR));
    double SR = max((vL[0] + cfL), (vR[0] + cfR));

    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);

    if (SR <= SL)
    {
        printf("231\n");
    }

    double SM00 = SM;
    double SR00 = SR;
    double SL00 = SL;
    double SM01, SR01, SL01;
    if ((SM00 >= SR00) || (SM00 <= SL00))
    {
        SL = min((vL[0] - cfL), (vR[0] - cfR));
        SR = max((vL[0] + cfL), (vR[0] + cfR));
        suR = SR - vR[0];
        suL = SL - vL[0];
        SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);
        SM01 = SM;
        SR01 = SR;
        SL01 = SL;
        if ((SM01 >= SR01) || (SM01 <= SL01))
        {
            printf("251\n");
        }
    }


    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double upt1 = (kv(u1) + kv(v1) + kv(w1)) / 2.0;
    double sbv1 = u1 * bx1 + v1 * by1 + w1 * bz1;

    double upt2 = (kv(u2) + kv(v2) + kv(w2)) / 2.0;
    double sbv2 = u2 * bx2 + v2 * by2 + w2 * bz2;

    double e1 = p1 / g1 + r1 * upt1 + b2L / 2.0;
    double e2 = p2 / g1 + r2 * upt2 + b2R / 2.0;

    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + ptL - kv(bL[0]);
    FL[2] = r1 * vL[0] * vL[1] - bL[0] * bL[1];
    FL[3] = r1 * vL[0] * vL[2] - bL[0] * bL[2];
    FL[4] = (e1 + ptL) * vL[0] - bL[0] * sbv1;
    FL[5] = 0.0;
    FL[6] = vL[0] * bL[1] - vL[1] * bL[0];
    FL[7] = vL[0] * bL[2] - vL[2] * bL[0];

    FR[0] = r2 * vR[0];
    FR[1] = r2 * vR[0] * vR[0] + ptR - kv(bR[0]);
    FR[2] = r2 * vR[0] * vR[1] - bR[0] * bR[1];
    FR[3] = r2 * vR[0] * vR[2] - bR[0] * bR[2];
    FR[4] = (e2 + ptR) * vR[0] - bR[0] * sbv2;
    FR[5] = 0.0;
    FR[6] = vR[0] * bR[1] - vR[1] * bR[0];
    FR[7] = vR[0] * bR[2] - vR[2] * bR[0];

    UL[0] = r1;
    UL[4] = e1;
    UR[0] = r2;
    UR[4] = e2;


    for (int ik = 0; ik < 3; ik++)
    {
        UL[ik + 1] = r1 * vL[ik];
        UL[ik + 5] = bL[ik];
        UR[ik + 1] = r2 * vR[ik];
        UR[ik + 5] = bR[ik];
    }

    for (int ik = 0; ik < 8; ik++)
    {
        UZ[ik] = (SR * UR[ik] - SL * UL[ik] + FL[ik] - FR[ik]) / (SR - SL);
    }

    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = r2 * suRm;
    double rzL = r1 * suLm;
    vzR[0] = SM;
    vzL[0] = SM;
    double ptzR = ptR + r2 * suR * (SM - vR[0]);
    double ptzL = ptL + r1 * suL * (SM - vL[0]);
    double ptz = (ptzR + ptzL) / 2.0;
    bzR[0] = UZ[5];
    bzL[0] = UZ[5];

    vzR[1] = UZ[2] / UZ[0];
    vzR[2] = UZ[3] / UZ[0];
    vzL[1] = vzR[1];
    vzL[2] = vzR[2];

    vzR[1] = vR[1] + UZ[5] * (bR[1] - UZ[6]) / suR / r2;
    vzR[2] = vR[2] + UZ[5] * (bR[2] - UZ[7]) / suR / r2;
    vzL[1] = vL[1] + UZ[5] * (bL[1] - UZ[6]) / suL / r1;
    vzL[2] = vL[2] + UZ[5] * (bL[2] - UZ[7]) / suL / r1;

    bzR[1] = UZ[6];
    bzR[2] = UZ[7];
    bzL[1] = bzR[1];
    bzL[2] = bzR[2];

    double sbvz = (UZ[5] * UZ[1] + UZ[6] * UZ[2] + UZ[7] * UZ[3]) / UZ[0];

    double ezR = e2 * suRm + (ptz * SM - ptR * vR[0] + UZ[5] * (sbv2 - sbvz)) / (SR - SM);
    double ezL = e1 * suLm + (ptz * SM - ptL * vL[0] + UZ[5] * (sbv1 - sbvz)) / (SL - SM);

    if (fabs(UZ[5]) < eps)
    {
        vzR[1] = vR[1];
        vzR[2] = vR[2];
        vzL[1] = vL[1];
        vzL[2] = vL[2];
        bzR[1] = bR[1] * suRm;
        bzR[2] = bR[2] * suRm;
        bzL[1] = bL[1] * suLm;
        bzL[2] = bL[2] * suLm;
    }
    UZL[0] = rzL;
    UZL[4] = ezL;
    UZR[0] = rzR;
    UZR[4] = ezR;

    for (int ik = 0; ik < 3; ik++)
    {
        UZL[ik + 1] = vzL[ik] * rzL;
        UZL[ik + 5] = bzL[ik];
        UZR[ik + 1] = vzR[ik] * rzR;
        UZR[ik + 5] = bzR[ik];
    }

    if (SL > wv)
    {
        Ps.x = FL[0] - wv * UL[0];
        Ps.y = FL[4] - wv * UL[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FL[ik] - wv * UL[ik];
        }
    }
    else if ( (SL <= wv) && (SM >= wv) )
    {
        Ps.x = FL[0] + SL * (rzL - r1) - wv * UZL[0];
        Ps.y = FL[4] + SL * (ezL - e1) - wv * UZL[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
        }
    }
    else if ((SM <= wv)&&(SR >= wv))
    {
        Ps.x = FR[0] + SR * (rzR - r2) - wv * UZR[0];
        Ps.y = FR[4] + SR * (ezR - e2) - wv * UZR[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
        }
    }
    else if (SR < wv)
    {
        Ps.x = FR[0] - wv * UR[0];
        Ps.y = FR[4] - wv * UR[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FR[ik] + - wv * UR[ik];
        }
    }
    else
    {
        printf("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n");
    }


    Pu.x = aco[0][0] * qv[0] + aco[0][1] * qv[1] + aco[0][2] * qv[2];
    Pu.y = aco[1][0] * qv[0] + aco[1][1] * qv[1] + aco[1][2] * qv[2];

    return time;
}

__device__ double HLLC_Aleksashov2(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double r1 = Ls.x;
    double p1 = Ls.y;
    double u1 = Lu.x;
    double v1 = Lu.y;

    double r2 = Rs.x;
    double p2 = Rs.y;
    double u2 = Ru.x;
    double v2 = Ru.y;



    // c------ - n_state = 2 - two - state(3 speed) HLLC(Contact Discontinuity)


    double ro = (r2 + r1) / 2.0;
    double ap = (p2 + p1) / 2.0;

    double aco[2][2];
    aco[0][0] = n1;
    aco[1][0] = n2;
    aco[0][1] = -n2;
    aco[1][1] = n1;

    //aco(1, 1) = al
    //aco(2, 1) = be
    //aco(3, 1) = ge

    double vL[2];
    double vR[2];

    vL[0] = aco[0][0] * u1 + aco[1][0] * v1;
    vL[1] = aco[0][1] * u1 + aco[1][1] * v1;
    vR[0] = aco[0][0] * u2 + aco[1][0] * v2;
    vR[1] = aco[0][1] * u2 + aco[1][1] * v2;

    if ((r1 <= eps) || (r2 <= eps) || (p1 <= 0) || (p2 <= 0) )
    {
        printf("EREREREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n");
    }

    double cL = __dsqrt_rn(ga * p1 / r1);
    double cR = __dsqrt_rn(ga * p2 / r2);
    double cC = __dsqrt_rn(ga * ap / ro);

    double SL, SR;

    //SL = min((vL[0] - cL), (vC1 - cC));
    //SR = max((vR[0] + cR), (vC1 + cC));

    SL = min((vL[0] - cL), (vR[0] - cR));
    SR = max((vL[0] + cL), (vR[0] + cR));

    double t = 10000000;
    t = min(t, krit * rad / max(fabs(SL), fabs(SR)));

    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = 0.0;
    if (fabs(suR * r2 - suL * r1) > 0)
    {
        SM = (suR * r2 * vR[0] - p2 + p1 - suL * r1 * vL[1]) / (suR * r2 - suL * r1);
    }

    if (SR < SL)
    {
        printf("12102022020,    ERROR in HLCC_Alexashov  \n");
    }

    double upt1 = (u1 * u1 + v1 * v1) / 2.0;
    double upt2 = (u2 * u2 + v2 * v2) / 2.0;
    double e1 = p1 / g1 + r1 * upt1;
    double e2 = p2 / g1 + r2 * upt2;
    double FL[4];
    double FR[4];
    double UL[4];
    double UR[4];

    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + p1;
    FL[2] = r1 * vL[0] * vL[1];
    FL[3] = (e1 + p1) * vL[0];

    FR[0] = r2 * vR[0];
    FR[1] = r2 * vR[0] * vR[0] + p2;
    FR[2] = r2 * vR[0] * vR[1];
    FR[3] = (e2 + p2) * vR[0];

    UL[0] = r1;
    UL[3] = e1;
    UR[0] = r2;
    UR[3] = e2;

    UL[1] = r1 * vL[0];
    UL[2] = r1 * vL[1];
    UR[1] = r2 * vR[0];
    UR[2] = r2 * vR[1];


    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = r2 * suRm;
    double rzL = r1 * suLm;

    double ptzR = p2 + r2 * suR * (SM - vR[0]);
    double ptzL = p1 + r1 * suL * (SM - vL[0]);
    double ptz = (ptzR + ptzL) / 2.0;
    double vzR[2];
    double vzL[2];

    vzR[0] = SM;
    vzL[0] = SM;
    vzR[1] = vR[1];
    vzL[1] = vL[1];

    double ezR = e2 * suRm + (ptz * SM - p2 * vR[0]) / (SR - SM);
    double ezL = e1 * suLm + (ptz * SM - p1 * vL[0]) / (SL - SM);

    double UZL[4];
    double UZR[4];

    UZL[0] = rzL;
    UZL[3] = ezL;
    UZR[0] = rzR;
    UZR[3] = ezR;

    for (int i = 1; i < 3; i++)
    {
        UZL[i] = vzL[i - 1] * rzL;
        UZR[i] = vzR[i - 1] * rzR;
    }

    double qv[2];

    if (SL > 0.0)
    {
        Ps.x = FL[0];
        Ps.y = FL[3];
        qv[0] = FL[1];
        qv[1] = FL[2];
    }
    else if ((SL <= 0.0) && (SM >= 0.0))
    {
        Ps.x = FL[0] + SL * (rzL - r1);
        Ps.y = FL[3] + SL * (ezL - e1);
        qv[0] = FL[1] + SL * (UZL[1] - UL[1]);
        qv[1] = FL[2] + SL * (UZL[2] - UL[2]);
    }
    else if ((SM <= 0.0) && (SR >= 0.0))
    {
        Ps.x = FR[0] + SR * (rzR - r2);
        Ps.y = FR[3] + SR * (ezR - e2);
        qv[0] = FR[1] + SR * (UZR[1] - UR[1]);
        qv[1] = FR[2] + SR * (UZR[2] - UR[2]);
    }
    else if (SR < 0.0)
    {
        Ps.x = FR[0];
        Ps.y = FR[3];
        qv[0] = FR[1];
        qv[1] = FR[2];
    }
    else
    {
        printf("21702022020,    ERROR in HLCC_Alexashov  \n");
        printf(" SL = %lf, SM = %lf, SR = %lf\n", SL, SM, SR);
        printf(" r1 = %lf, p1 = %lf, u1 = %lf, v1 = %lf\n", r1, p1, u1, v1);
        printf(" r2 = %lf, p2 = %lf, u2 = %lf, v2 = %lf\n", r2, p2, u2, v2);
        printf(" vl[0] = %lf, cL = %lf, vR[0] = %lf, cR = %lf\n", vL[0], cL, vR[0], cR);
        /*SL = min((vL[0] - cL), (vR[0] - cR));
        SR = max((vL[0] + cL), (vR[0] + cR));*/
    }

    Pu.x = aco[0][0] * qv[0] + aco[0][1] * qv[1];
    Pu.y = aco[1][0] * qv[0] + aco[1][1] * qv[1];

    return t;
}

__device__ double HLLC_Korolkov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double ro_L = Ls.x;
    double p_L = Ls.y;
    double v1_L = Lu.x;
    double v2_L = Lu.y;

    double ro_R = Rs.x;
    double p_R = Rs.y;
    double v1_R = Ru.x;
    double v2_R = Ru.y;

    double e_L, e_R;
    double Vkv_L, Vkv_R;
    double c_L, c_R;

    Vkv_L = v1_L * v1_L + v2_L * v2_L;
    Vkv_R = v1_R * v1_R + v2_R * v2_R;

    c_L = __dsqrt_rn(ggg * p_L / ro_L);
    c_R = __dsqrt_rn(ggg * p_R / ro_R);
    e_L = p_L / (ggg - 1.0) + ro_L * Vkv_L / 2.0;  /// ������ ������� �����
    e_R = p_R / (ggg - 1.0) + ro_R * Vkv_R / 2.0;  /// ������ ������� ������

    double Vn_L = v1_L * n1 + v2_L * n2;
    double Vn_R = v1_R * n1 + v2_R * n2;

    double D_L = min(Vn_L, Vn_R) - max(c_L, c_R);
    double D_R = max(Vn_L, Vn_R) + max(c_L, c_R);
    /*double D_L = min(Vn_L - c_L, Vn_R - c_R);
    double D_R = max(Vn_L + c_L, Vn_R + c_R);*/
    double t = 10000000;
    t = min(t, krit * rad / max(fabs(D_L), fabs(D_R)));

    double fx1 = ro_L * v1_L;
    double fx2 = ro_L * v1_L * v1_L + p_L;
    double fx3 = ro_L * v1_L * v2_L;
    double fx5 = (e_L + p_L) * v1_L;

    double fy1 = ro_L * v2_L;
    double fy2 = ro_L * v1_L * v2_L;
    double fy3 = ro_L * v2_L * v2_L + p_L;
    double fy5 = (e_L + p_L) * v2_L;

    double fl_1 = fx1 * n1 + fy1 * n2;
    double fl_2 = fx2 * n1 + fy2 * n2;
    double fl_3 = fx3 * n1 + fy3 * n2;
    double fl_5 = fx5 * n1 + fy5 * n2;

    if (D_L > Omega)
    {
        Ps.x = fl_1; /// ����� ����� ��������� �� ������� ����� � ��� �� �������
        Pu.x = fl_2;
        Pu.y = fl_3;
        Ps.y = fl_5;
        return t;
    }

    double hx1 = ro_R * v1_R;
    double hx2 = ro_R * v1_R * v1_R + p_R;
    double hx3 = ro_R * v1_R * v2_R;
    double hx5 = (e_R + p_R) * v1_R;

    double hy1 = ro_R * v2_R;
    double hy2 = ro_R * v1_R * v2_R;
    double hy3 = ro_R * v2_R * v2_R + p_R;
    double hy5 = (e_R + p_R) * v2_R;

    double fr_1 = hx1 * n1 + hy1 * n2;
    double fr_2 = hx2 * n1 + hy2 * n2;
    double fr_3 = hx3 * n1 + hy3 * n2;
    double fr_5 = hx5 * n1 + hy5 * n2;

    if (D_R < Omega)
    {
        Ps.x = fr_1; /// ����� ����� ��������� �� ������� ����� � ��� �� �������
        Pu.x = fr_2;
        Pu.y = fr_3;
        Ps.y = fr_5;
        return t;
    }

    double u_L = Vn_L;
    double u_R = Vn_R;

    double D_C = ((D_R - u_R) * ro_R * u_R - (D_L - u_L) * ro_L * u_L - p_R + p_L) / ((D_R - u_R) * ro_R - (D_L - u_L) * ro_L);

    double roro_L = ro_L * ((D_L - u_L) / (D_L - D_C));
    double roro_R = ro_R * ((D_R - u_R) / (D_R - D_C));

    /// ������� �������� � ����������� ������� (��� ���������� ����� � ������)
    double P_T = (p_L * ro_R * (u_R - D_R) - p_R * ro_L * (u_L - D_L) - ro_L * ro_R * (u_L - D_L) * (u_R - D_R) * (u_R - u_L)) / (ro_R * (u_R - D_R) - ro_L * (u_L - D_L));

    if (D_L <= Omega && D_C >= Omega)  /// ������ �� ������ ������� (�����)
    {
        double Vx = v1_L + (D_C - Vn_L) * n1;
        double Vy = v2_L + (D_C - Vn_L) * n2;

        double ee_L = P_T / (ggg - 1.0) + roro_L * (Vx * Vx + Vy * Vy) / 2.0;
        //double ee_L = e_L - ((P_T - p_L)/2.0)*(1/roro_L - 1/ro_L);
        /*double ee_L = ((D_L - u_L) * e_L - p_L * u_L + P_T * D_C) / (D_L - D_C);*/

        double dq1 = roro_L - ro_L;
        double dq2 = roro_L * Vx - ro_L * v1_L;
        double dq3 = roro_L * Vy - ro_L * v2_L;
        double dq5 = ee_L - e_L;

        Ps.x = D_L * dq1 + fl_1; /// ����� ����� ��������� �� ������� ����� � ��� �� �������
        Pu.x = D_L * dq2 + fl_2;
        Pu.y = D_L * dq3 + fl_3;
        Ps.y = D_L * dq5 + fl_5;
        return t;
    }
    else if (D_R >= Omega && D_C <= Omega)  /// ������ �� ������ ������� (������)
    {
        double Vx = v1_R + (D_C - Vn_R) * n1;
        double Vy = v2_R + (D_C - Vn_R) * n2;

        double ee_R = P_T / (ggg - 1.0) + roro_R * (Vx * Vx + Vy * Vy) / 2.0;
        /*double ee_R = ((D_R - u_R) * e_R - p_R * u_R + P_T * D_C) / (D_R - D_C);*/

        double dq1 = roro_R - ro_R;
        double dq2 = roro_R * Vx - ro_R * v1_R;
        double dq3 = roro_R * Vy - ro_R * v2_R;
        double dq5 = ee_R - e_R;

        Ps.x = D_R * dq1 + fr_1; /// ����� ����� ��������� �� ������� ����� � ��� �� �������
        Pu.x = D_R * dq2 + fr_2;
        Pu.y = D_R * dq3 + fr_3;
        Ps.y = D_R * dq5 + fr_5;
        return t;
    }
    return t;
}

__device__ double HLL(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double ro_L = Ls.x;
    double p_L = Ls.y;
    double v1_L = Lu.x;
    double v2_L = Lu.y;

    double ro_R = Rs.x;
    double p_R = Rs.y;
    double v1_R = Ru.x;
    double v2_R = Ru.y;

    double e_L, e_R;
    double Vkv_L, Vkv_R;
    double c_L, c_R;

    Vkv_L = v1_L * v1_L + v2_L * v2_L;
    Vkv_R = v1_R * v1_R + v2_R * v2_R;
    if (ro_L <= 0)
    {
        c_L = 0.0;
    }
    else
    {
        c_L = sqrt(ggg * p_L / ro_L);
    }

    if (ro_R <= 0)
    {
        c_R = 0.0;
    }
    else
    {
        c_R = sqrt(ggg * p_R / ro_R);
    }
    e_L = p_L / (ggg - 1.0) + ro_L * Vkv_L / 2.0;  /// ������ ������� �����
    e_R = p_R / (ggg - 1.0) + ro_R * Vkv_R / 2.0;  /// ������ ������� ������

    double Vn_L = v1_L * n1 + v2_L * n2;
    double Vn_R = v1_R * n1 + v2_R * n2;
    double D_L = my_min(Vn_L, Vn_R) - my_max(c_L, c_R);
    double D_R = my_max(Vn_L, Vn_R) + my_max(c_L, c_R);
    double t = 10000000;
    t = my_min(t, krit * rad / my_max(fabs(D_L), fabs(D_R)));

    double fx1 = ro_L * v1_L;
    double fx2 = ro_L * v1_L * v1_L + p_L;
    double fx3 = ro_L * v1_L * v2_L;
    double fx5 = (e_L + p_L) * v1_L;

    double fy1 = ro_L * v2_L;
    double fy2 = ro_L * v1_L * v2_L;
    double fy3 = ro_L * v2_L * v2_L + p_L;
    double fy5 = (e_L + p_L) * v2_L;

    double fl_1 = fx1 * n1 + fy1 * n2;
    double fl_2 = fx2 * n1 + fy2 * n2;
    double fl_3 = fx3 * n1 + fy3 * n2;
    double fl_5 = fx5 * n1 + fy5 * n2;

    /*double U_L1 = ro_L;
    double U_L2 = ro_L * v1_L;
    double U_L3 = ro_L * v2_L;
    double U_L5 = e_L;*/

    if (D_L > Omega)
    {
        Ps.x = fl_1; /// ����� ����� ��������� �� ������� ����� � ��� �� �������
        Pu.x = fl_2;
        Pu.y = fl_3;
        Ps.y = fl_5;
        return t;
    }
    else
    {
        double hx1 = ro_R * v1_R;
        double hx2 = ro_R * v1_R * v1_R + p_R;
        double hx3 = ro_R * v1_R * v2_R;
        double hx5 = (e_R + p_R) * v1_R;

        double hy1 = ro_R * v2_R;
        double hy2 = ro_R * v1_R * v2_R;
        double hy3 = ro_R * v2_R * v2_R + p_R;
        double hy5 = (e_R + p_R) * v2_R;

        double fr_1 = hx1 * n1 + hy1 * n2;
        double fr_2 = hx2 * n1 + hy2 * n2;
        double fr_3 = hx3 * n1 + hy3 * n2;
        double fr_5 = hx5 * n1 + hy5 * n2;

        /*double U_R1 = ro_R;
        double U_R2 = ro_R * v1_R;
        double U_R3 = ro_R * v2_R;
        double U_R5 = e_R;*/

        if (D_R < Omega)
        {
            Ps.x = fr_1; /// ����� ����� ��������� �� ������� ����� � ��� �� �������
            Pu.x = fr_2;
            Pu.y = fr_3;
            Ps.y = fr_5;
            return t;
        }
        else
        {
            double dq1 = ro_R - ro_L;
            double dq2 = ro_R * v1_R - ro_L * v1_L;
            double dq3 = ro_R * v2_R - ro_L * v2_L;
            double dq5 = e_R - e_L;

            //double U1 = (D_R * U_R1 - D_L * U_L1 - fr_1 + fl_1) / (D_R - D_L);
            //double U2 = (D_R * U_R2 - D_L * U_L2 - fr_2 + fl_2) / (D_R - D_L);
            //double U3 = (D_R * U_R3 - D_L * U_L3 - fr_3 + fl_3) / (D_R - D_L);
            //double U5 = (D_R * U_R5 - D_L * U_L5 - fr_5 + fl_5) / (D_R - D_L);


            Ps.x = (D_R * fl_1 - D_L * fr_1 + D_L * D_R * dq1) / (D_R - D_L); /// ����� ����� ��������� �� ������� ����� � ��� �� �������
            Pu.x = (D_R * fl_2 - D_L * fr_2 + D_L * D_R * dq2) / (D_R - D_L);
            Pu.y = (D_R * fl_3 - D_L * fr_3 + D_L * D_R * dq3) / (D_R - D_L);
            Ps.y = (D_R * fl_5 - D_L * fr_5 + D_L * D_R * dq5) / (D_R - D_L);
            return t;
        }
    }
}

__device__ void predictor(const double *Q, const double* DX, const double* DY, double* QQ, double step_time, double x, double y)
{
    double P[7];
    double2 PS = { 0.0, 0.0 };
    double2 PU = { 0.0, 0.0 };
    double3 PB = { 0.0, 0.0, 0.0 };
    double Pdiv = 0.0;
    double ro_, p_;

    ro_ = Q[0] - DX[0] / 2.0;
    p_ = Q[1] - DX[1] / 2.0;
    if (ro_ <= 0.0)
    {
        ro_ = Q[0];
    }
    if (p_ <= 0.0)
    {
        p_ = Q[1];
    }
    POTOK_Korolkov(ro_, 0.0, p_, Q[2] - DX[2] / 2.0, Q[3] - DX[3] / 2.0, 0.0,//
        Q[4] - DX[4] / 2.0, Q[5] - DX[5] / 2.0, Q[6] - DX[6] / 2.0, P, -1.0, 0.0, 0.0);
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;

    ro_ = Q[0] + DX[0] / 2.0;
    p_ = Q[1] + DX[1] / 2.0;
    if (ro_ <= 0.0)
    {
        ro_ = Q[0];
    }
    if (p_ <= 0.0)
    {
        p_ = Q[1];
    }
    POTOK_Korolkov(ro_, 0.0, p_, Q[2] + DX[2] / 2.0, Q[3] + DX[3] / 2.0, 0.0,//
        Q[4] + DX[4] / 2.0, Q[5] + DX[5] / 2.0, Q[6] + DX[6] / 2.0, P, 1.0, 0.0, 0.0);
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;

    ro_ = Q[0] + DY[0] / 2.0;
    p_ = Q[1] + DY[1] / 2.0;
    if (ro_ <= 0.0)
    {
        ro_ = Q[0];
    }
    if (p_ <= 0.0)
    {
        p_ = Q[1];
    }
    POTOK_Korolkov(ro_, 0.0, p_, Q[2] + DY[2] / 2.0, Q[3] + DY[3] / 2.0, 0.0,//
        Q[4] + DY[4] / 2.0, Q[5] + DY[5] / 2.0, Q[6] + DY[6] / 2.0, P, 0.0, 1.0, 0.0);
    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;

    ro_ = Q[0] - DY[0] / 2.0;
    p_ = Q[1] - DY[1] / 2.0;
    if (ro_ <= 0.0)
    {
        ro_ = Q[0];
    }
    if (p_ <= 0.0)
    {
        p_ = Q[1];
    }
    POTOK_Korolkov(ro_, 0.0, p_, Q[2] - DY[2] / 2.0, Q[3] - DY[3] / 2.0, 0.0,//
        Q[4] - DY[4] / 2.0, Q[5] - DY[5] / 2.0, Q[6] - DY[6] / 2.0, P, 0.0, -1.0, 0.0);
    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;

    double dV = dx * dy;

    QQ[0] = Q[0] - step_time * PS.x / dV - step_time * Q[0] * Q[3] / y;
    if (QQ[0] <= 0)
    {
        printf("Problemsssss 84745377! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, QQ[0], step_time, Q[0]);
        QQ[0] = Q[0];
    }
    QQ[2] = (Q[0] * Q[2] - step_time * (PU.x) / dV - step_time * (Q[0] * Q[2] * Q[3] - Q[4] * Q[5] / cpi4) / y) / QQ[0];
    QQ[3] = (Q[0] * Q[3] - step_time * (PU.y) / dV - step_time * (Q[0] * Q[3] * Q[3] + (kv(Q[6]) - kv(Q[5])) / cpi4) / y) / QQ[0];
    QQ[4] = (Q[4] - step_time * (PB.x) / dV - step_time * (Q[3] * Q[4] - Q[5] * Q[2]) / y);
    QQ[5] = (Q[5] - step_time * (PB.y) / dV);
    QQ[6] = (Q[6] - step_time * (PB.z) / dV);
    QQ[1] = (U8(Q[0], Q[1], Q[2], Q[3], 0.0, Q[4], Q[5], Q[6]) - step_time * (PS.y)//
        / dV - step_time * (((U8(Q[0], Q[1], Q[2], Q[3], 0.0, Q[4], Q[5], Q[6]) + Q[1] + kvv(Q[4], Q[5], Q[6]) / cpi8) * Q[3] - Q[5] * skk(Q[2], Q[3], 0.0, Q[4], Q[5], Q[6]) / cpi4) / y) //
        - 0.5 * QQ[0] * kvv(QQ[2], QQ[3], 0.0) - kvv(QQ[4], QQ[5], QQ[6]) / cpi8) * (ggg - 1.0);
    if (QQ[1] <= 0)
    {
        QQ[1] = 0.000001;
    }
}

__global__ void funk_time(double* T, double* T_do, double* TT, int* i)
{
    *T_do = *T;
    *TT = *TT + *T_do;
    *T = 10000000;
    *i = *i + 1;
    if (*i % 5000 == 0)
    {
        printf("i = %d,  TT = %lf \n", *i, *TT);
    }
    return;
}


__global__ void Ker_Dekard(double2* s, double2* u, double2* s2, double2* u2, double* T, double* T_do, int method)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // ���������� ������ ������� ������ (�������� ������)
    int n = index % N;                                   // ����� ������ �� x (�� 0)
    int m = (index - n) / N;                             // ����� ������ �� y (�� 0)
    double y = y_min + m * (y_max) / (M);
    double x = x_min + n * (x_max - x_min) / (N - 1);
    //double dist = __dsqrt_rn(x * x + y * y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // ���������� ���� ������� � ����� ������
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // ������� �������
    double tmin = 1000;

    s_1 = s[index];
    u_1 = u[index];
    if ((n == N - 1)||(m == M-1)) // Ƹ����� ��������� �������
    {
        // � ���� ������� �������� ���������� ������������� � �� �������� � �������� �������)
        //s2[index] = s_1;
        //u2[index] = u_1;
        return;
    }
    s_2 = s[(m)*N + n + 1];
    u_2 = u[(m)*N + n + 1];

    if ((n == 0))
    {
        s_4 = s_1;
        u_4 = u_1;
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
    }

    if ((m == 0))
    {
        s_3 = s_1;
        u_3 = u_1;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
    }

    if ((m == M - 1))
    {
        s_5 = s_1;
        u_5 = u_1;
    }
    else
    {
        s_5 = s[(m + 1) * N + (n)];
        u_5 = u[(m + 1) * N + (n)];
    }


    if (method == 0)
    {
        tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dy));
        tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dx));
        tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dy));
        tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dx));
    }
    else if (method == 1)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dy));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dx));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dy));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dx));
        }
    }


    if (*T > tmin)
    {
        //__threadfence();
        *T = tmin;
    }

    double2 PS = { 0,0 };
    double2 PU = { 0,0 };

    /*if ((x > 400) &&( x < 405) && (y < 5))
    {
        printf("%lf, %lf, %lf, %lf, %lf\n", Ps12.x, Ps13.x, Ps14.x, Ps15.x, Ps12.x* dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx);
    }*/

    PS.x = Ps12.x * dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx;
    PS.y = Ps12.y * dy + Ps13.y * dx + Ps14.y * dy + Ps15.y * dx;
    PU.x = Pu12.x * dy + Pu13.x * dx + Pu14.x * dy + Pu15.x * dx;
    PU.y = Pu12.y * dy + Pu13.y * dx + Pu14.y * dy + Pu15.y * dx;

    double dV = dx * dy;

    s2[index].x = s[index].x - (*T_do / dV) * PS.x;
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf\n", x, y, s2[index].x);
        s2[index].x = 0.0001;
    }
    u2[index].x = (s[index].x * u[index].x - (*T_do / dV) * PU.x) / s2[index].x;
    u2[index].y = (s[index].x * u[index].y - (*T_do / dV) * PU.y) / s2[index].x;
    s2[index].y = (((s[index].y / (ggg - 1) + s[index].x * (u[index].x * u[index].x + u[index].y * u[index].y) * 0.5) - (*T_do / dV) * PS.y) - //
        0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);
    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }
}

__global__ void add2(double2* s, double2* u, double3* b, double2* s2, double2* u2, double3* b2, double* T, double* T_do, int i, int method)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // ���������� ������ ������� ������ (�������� ������)
    int n = index % N;                                   // ����� ������ �� x (�� 0)
    int m = (index - n) / N;                             // ����� ������ �� y (�� 0)
    double y = y_min + m * dy; // (y_max - y_min) / (M - 1);
    double x = x_min + n * dx; // (x_max - x_min) / (N - 1);
    double dist = sqrt(x * x + y * y);
    int sign_y = 1;
    if (y < 0.0)
    {
        sign_y = -1;
    }
    y = fabs(y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // ���������� ���� ������� � ����� ������
    double3 b_1, b_2, b_3, b_4, b_5;
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // ������� �������
    double3 Pb12 = { 0.0, 0.0, 0.0 }, Pb13 = { 0.0, 0.0, 0.0 }, Pb14 = { 0.0, 0.0, 0.0 }, Pb15 = { 0.0, 0.0, 0.0 };
    double tmin = 1000;
    double P[8];
    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

    if (index < 0 || index > N * M - 1)
    {
        printf("Error index = %d \n", index);
    }

    double n1, n2, nn;

    s_1 = s[index];
    u_1 = u[index];
    b_1 = b[index];


   /* double ss = 3.0;
    if (i > 20000)
    {
        ss = 2.0;
    }*/

    if ( (dist < ddist))  // Ƹ����� ��������� �������
    {
        // � ���� ������� �������� ���������� ������������� � �� �������� � �������� �������)
        s2[index] = s_1;
        u2[index] = u_1;
        b2[index] = b_1;
        return;
    }

    //if ((y > 16.0)&&(i%100 == 0))  // Ƹ����� ��������� �������
    //{
    //    // � ���� ������� �������� ���������� ������������� � �� �������� � �������� �������)
    //    s2[index] = { 1.0, 1.0 / ggg };
    //    u2[index] = { 0.0, 0.0 };
    //    b2[index] = b_1;
    //    return;
    //}


    if ((m == M - 1))
    {
        /*s_5 = s_1;
        u_5 = u_1;
        if (u_5.y < 0)
        {
            u_5.y = 0.001;
        }
        b_5 = b_1;*/
        s_5 = { 1.0, 1.0 / ggg };
        u_5 = { 0.0, 0.0 };
        b_5 = b_1;
    }
    else
    {
        s_5 = s[(m + 1)*N + n];
        u_5 = u[(m + 1)*N + n];
        b_5 = b[(m + 1) * N + n];
    }

    if ((n == N - 1))
    {
        s_2 = s_1;
        u_2 = u_1;
        if (u_2.x < 0)
        {
            u_2.x = 0.01;
        }
        b_2 = b_1;
    }
    else
    {
        s_2 = s[(m) * N + n + 1];
        u_2 = u[(m) * N + n + 1];
        b_2 = b[(m) * N + n + 1];
    }

    if (n == 0)
    {
        /*s_5 = { 1.0, 1.0 / ggg };
        u_5 = { 0.0, 0.0 };
        b_5 = b_1;*/

        s_4 = s_1;
        u_4.y = u_1.y;
        u_4.x = -u_1.x;
        b_4.x = -b_1.x;
        b_4.y = -b_1.y;
        b_4.z = b_1.z;
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
        b_4 = b[(m)*N + n - 1];
    }

    if ((m == 0))
    {
        if (false)
        {
            s_5 = { 1.0, 1.0 / ggg };
            u_5 = { 0.0, 0.0 };
            b_5 = b_1;
        }
        else
        {
            s_3 = s_1;
            u_3.x = u_1.x;
            u_3.y = -u_1.y;
            //u_3.y = 0.0;
            //b_3.x = -b_1.x;
            b_3.x = b_1.x;
            b_3.y = b_1.y;
            b_3.z = -b_1.z;
            //b_3.z = 0.0;
        }
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
        b_3 = b[(m - 1) * N + (n)];
    }

    double2 sss;
    double3 bbb;

    if (sign_y == -1)
    {
        sss = s_5;
        s_5 = s_3;
        s_3 = sss;

        sss = u_5;
        u_5 = u_3;
        u_3 = sss;

        bbb = b_5;
        b_5 = b_3;
        b_3 = bbb;

        u_1.y = u_1.y * sign_y;
        u_2.y = u_2.y * sign_y;
        u_3.y = u_3.y * sign_y;
        u_4.y = u_4.y * sign_y;
        u_5.y = u_5.y * sign_y;

        b_1.y = b_1.y * sign_y;
        b_2.y = b_2.y * sign_y;
        b_3.y = b_3.y * sign_y;
        b_4.y = b_4.y * sign_y;
        b_5.y = b_5.y * sign_y;

        b_1.z = b_1.z * sign_y;
        b_2.z = b_2.z * sign_y;
        b_3.z = b_3.z * sign_y;
        b_4.z = b_4.z * sign_y;
        b_5.z = b_5.z * sign_y;
    }

    

    double Q = 1.0;
    double PQ = 0.0;
    double2 PS = { 0.0, 0.0 };
    double2 PU = { 0.0, 0.0 };
    double3 PB = { 0.0, 0.0, 0.0 };
    double Pdiv = 0.0;

    
    tmin = my_min(tmin, HLLDQ_Korolkov(s_1.x, Q, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z, s_2.x, Q, s_2.y, //
        u_2.x, u_2.y, 0.0, b_2.x, b_2.y, b_2.z, P, PQ, 1.0, 0.0, 0.0, dx, method, x, y));
    
    
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;
    Pdiv = Pdiv + dy * (b_1.x + b_2.x) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

    
    tmin = my_min(tmin, HLLDQ_Korolkov(s_1.x, Q, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z, s_3.x, Q, s_3.y, //
        u_3.x, u_3.y, 0.0, b_3.x, b_3.y, b_3.z, P, PQ, 0.0, -1.0, 0.0, dy, method, x, y));
    
    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;
    Pdiv = Pdiv - dx * (b_1.y + b_3.y) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
    
    tmin = my_min(tmin, HLLDQ_Korolkov(s_1.x, Q, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z, s_4.x, Q, s_4.y, //
        u_4.x, u_4.y, 0.0, b_4.x, b_4.y, b_4.z, P, PQ, -1.0, 0.0, 0.0, dx, method, x, y));
    
    
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;
    Pdiv = Pdiv - dy * (b_1.x + b_4.x) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
    
    tmin = my_min(tmin, HLLDQ_Korolkov(s_1.x, Q, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z, s_5.x, Q, s_5.y, //
        u_5.x, u_5.y, 0.0, b_5.x, b_5.y, b_5.z, P, PQ, 0.0, 1.0, 0.0, dy, method, x, y));
    
    
    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;
    Pdiv = Pdiv + dx * (b_1.y + b_5.y) / 2.0;
        
    //printf("%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,\n", PS.x, PS.y, PU.x, PU.y, PB.x, PB.y, PB.z, Pdiv);

    if (*T > tmin)
    {
        *T = tmin;
    }

    double dV = dx * dy;


    Pdiv = Pdiv + dV * b_1.y / y;
    //Pdiv = 0.0;


    s2[index].x = s_1.x - *T_do * PS.x / dV - *T_do * s_1.x * u_1.y / y;
    //s2[index].x = s_1.x - (*T_do / dV) * PS.x - *T_do * s_1.x * u_1.y / y;
    //s2[index].x = s_1.x - (*T_do / dV) * PS.x;   // � ���������� �����������
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, s2[index].x, *T_do, s_1.x);
        s2[index].x = s_1.x;
    }

    //u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * (PU.x + (b_1.x/cpi4)*Pdiv ) - *T_do * (s_1.x * u_1.y * u_1.x) / y) / s2[index].x;
    //u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * (PU.y + (b_1.y / cpi4) * Pdiv) - *T_do * s_1.x * u_1.y * u_1.y / y) / s2[index].x;
    //b2[index].x = (b_1.x - *T_do * (PB.x + u_1.x * Pdiv) / dV) - *T_do * (u_1.y*b_1.x - u_1.x * b_1.y)/y;
    //b2[index].y = (b_1.y - *T_do * (PB.y + u_1.y * Pdiv) / dV);
    //b2[index].z = (b_1.x - *T_do * (PB.z) / dV) - *T_do * (u_1.y * b_1.z) / y;

    //s2[index].y = ( ((s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * (PS.y + //
    //    (skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) / cpi4) * Pdiv) - //
    //    *T_do * u_1.y * (ggg * s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) / y) - //
    //    0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y) - kvv(b_1.x, b_1.y, b_1.z) / cpi8 ) * (ggg - 1);


    //s2[index].x = s_1.x - *T_do * PS.x / dV - *T_do * s_1.x * u_1.y / y;
    u2[index].x = (s_1.x * u_1.x - *T_do * (PU.x + (b_1.x / cpi4) * Pdiv) / dV  - *T_do * (s_1.x * u_1.x * u_1.y - b_1.x * b_1.y /cpi4)/y ) / s2[index].x;
    /*if (u2[index].x < 0.0)
    {
        u2[index].x = 0.0;
    }*/
    u2[index].y = (s_1.x * u_1.y - *T_do * (PU.y + (b_1.y / cpi4) * Pdiv) / dV - *T_do * (s_1.x * u_1.y * u_1.y + (kv(b_1.z) - kv(b_1.y)) / cpi4) / y ) / s2[index].x;
    b2[index].x = (b_1.x - *T_do * (PB.x + u_1.x * Pdiv) / dV - *T_do*(u_1.y * b_1.x - b_1.y * u_1.x)/y);
    b2[index].y = (b_1.y - *T_do * (PB.y + u_1.y * Pdiv) / dV);
    b2[index].z = (b_1.z - *T_do * (PB.z) / dV );
    s2[index].y = (U8(s_1.x, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) - *T_do * (PS.y + (skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) / cpi4) * Pdiv)//
        / dV - *T_do * ( ( (U8(s_1.x, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) + s_1.y + kvv(b_1.x, b_1.y, b_1.z) / cpi8)* u_1.y - b_1.y * skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z)/cpi4)/ y) //
        - 0.5 * s2[index].x * kvv(u2[index].x, u2[index].y, 0.0) - kvv(b2[index].x, b2[index].y, b2[index].z) / cpi8) * (ggg - 1.0);


    if (sign_y < 0.0)
    {
        u2[index].y = u2[index].y * sign_y;
        b2[index].y = b2[index].y * sign_y;
        b2[index].z = b2[index].z * sign_y;
    }


    //� ���������� �������
    //u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * PU.x ) / s2[index].x;
    //u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * PU.y) / s2[index].x;

    //s2[index].y = ( ( (s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * PS.y ) - //
    //    0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);

    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }
}

__global__ void Kernel_TVD(double2* s, double2* u, double3* b, double2* s2, double2* u2, double3* b2, double* T, double* T_do, int i, int method)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // ���������� ������ ������� ������ (�������� ������)
    int n = index % N;                                   // ����� ������ �� x (�� 0)
    int m = (index - n) / N;                             // ����� ������ �� y (�� 0)
    double y = y_min + m * dy; // (y_max) / (M);
    double x = x_min + n * dx; // (x_max) / (N);  //(x_max - x_min) / (N - 1);
    double dist = __dsqrt_rn(x * x + y * y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // ���������� ���� ������� � ����� ������
    double3 b_1, b_2, b_3, b_4, b_5;
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // ������� �������
    double3 Pb12 = { 0.0, 0.0, 0.0 }, Pb13 = { 0.0, 0.0, 0.0 }, Pb14 = { 0.0, 0.0, 0.0 }, Pb15 = { 0.0, 0.0, 0.0 };
    double tmin = 1000;

    s_1 = s[index];
    u_1 = u[index];
    b_1 = b[index];
    //if ((n == N - 1) || (m == M - 1) || (dist < 110)) // Ƹ����� ��������� �������
    if ((dist < ddist))  // Ƹ����� ��������� �������
    {
        // � ���� ������� �������� ���������� ������������� � �� �������� � �������� �������)
        s2[index] = s_1;
        u2[index] = u_1;
        b2[index] = b_1;
        return;
    }
    s_2 = s[(m)*N + n + 1];
    u_2 = u[(m)*N + n + 1];

    if((m == M - 1))
    {
        s_5 = { 1.0, 1.0 / ggg };
        u_5 = { 0.0, 0.0 };
        /*if (u_5.y < 0)
        {
            u_5.y = 0.01;
        }*/
        b_5 = { 0.0, 0.0 };  // b_1
    }
    else
    {
        s_5 = s[(m + 1) * N + n];
        u_5 = u[(m + 1) * N + n];
        b_5 = b[(m + 1) * N + n];
    }

    if ((n == N - 1))
    {
        s_2 = s_1;
        u_2 = u_1;
        if (u_2.x < 0)
        {
            u_2.x = 0.01;
        }
        b_2 = b_1;
    }
    else
    {
        s_2 = s[(m)*N + n + 1];
        u_2 = u[(m)*N + n + 1];
        b_2 = b[(m)*N + n + 1];
    }

    if (n == 0)
    {
        s_4 = s_1;
        u_4.y = u_1.y;
        u_4.x = -u_1.x;
        b_4.x = -b_1.x;
        b_4.y = -b_1.y;
        b_4.z = b_1.z;
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
        b_4 = b[(m)*N + n - 1];
    }

    if ((m == 0))
    {
        s_3 = s_1;
        u_3.x = u_1.x;
        u_3.y = -u_1.y;
        //b_3.x = -b_1.x;
        b_3.x = b_1.x;
        b_3.y = b_1.y;
        b_3.z = -b_1.z;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
        b_3 = b[(m - 1) * N + (n)];
    }

    double2 s12 = { 0.0 ,0.0 };
    double2 s13 = { 0.0 ,0.0 };
    double2 s14 = { 0.0 ,0.0 };
    double2 s15 = { 0.0 ,0.0 };
    double2 u12 = { 0.0 ,0.0 };
    double2 u13 = { 0.0 ,0.0 };
    double2 u14 = { 0.0 ,0.0 };
    double2 u15 = { 0.0 ,0.0 };
    double3 b12 = { 0.0 ,0.0 };
    double3 b13 = { 0.0 ,0.0 };
    double3 b14 = { 0.0 ,0.0 };
    double3 b15 = { 0.0 ,0.0 };
    double2 s21 = { 0.0 ,0.0 };
    double2 s31 = { 0.0 ,0.0 };
    double2 s41 = { 0.0 ,0.0 };
    double2 s51 = { 0.0 ,0.0 };
    double2 u21 = { 0.0 ,0.0 };
    double2 u31 = { 0.0 ,0.0 };
    double2 u41 = { 0.0 ,0.0 };
    double2 u51 = { 0.0 ,0.0 };
    double3 b21 = { 0.0 ,0.0 };
    double3 b31 = { 0.0 ,0.0 };
    double3 b41 = { 0.0 ,0.0 };
    double3 b51 = { 0.0 ,0.0 };
    double A = 0, B = 0;
    // ��������� ���������� �������-�������
    if (n >= N - 2)
    {
        s21 = s_2;
        u21 = u_2;
        if (u21.x < 0)
        {
            u21.x = 0.01;
        }
        b21 = b_2;
    }
    else
    {
        s21 = s[(m)*N + (n + 2)];
        u21 = u[(m)*N + (n + 2)];
        b21 = b[(m)*N + (n + 2)];
    }

    if (m >= M - 2) // ������ ������ ������� ��� ������
    {
        /*s51 = s_5;
        u51 = u_5;
        b51 = b_5;*/
        s51 = { 1.0, 1.0 / ggg };
        u51 = { 0.0, 0.0 };
        b51 = { 0.0, 0.0 };  
    }
    else
    {
        s51 = s[(m + 2) * N + (n)];
        u51 = u[(m + 2) * N + (n)];
        b51 = b[(m + 2) * N + (n)];
    }


    if (n == 0)
    {
        s41 = s_2;
        u41.y = u_2.y;
        u41.x = -u_2.x;
        b41.x = -b_2.x;
        b41.y = -b_2.y;
        b41.z = b_2.z;
    }
    else if (n == 1)
    {
        s41 = s_2;
        u41.y = u_2.y;
        u41.x = -u_2.x;
        b41.x = -b_2.x;
        b41.y = -b_2.y;
        b41.z = b_2.z;
    }
    else
    {
        s41 = s[(m)*N + (n - 2)];
        u41 = u[(m)*N + (n - 2)];
        b41 = b[(m)*N + (n - 2)];
    }


    if (m == 1)
    {
        s31 = s_5;
        u31.x = u_5.x;
        u31.y = -u_5.y;
        b31.x = b_5.x;
        b31.y = b_5.y;
        b31.z = -b_5.z;
    }
    else if (m == 0)
    {
        s31 = s_3;
        u31.x = u_3.x;
        u31.y = -u_3.y;
        b31.x = b_3.x;
        b31.y = b_3.y;
        b31.z = -b_3.z;
    }
    else
    {
        s31 = s[(m - 2) * N + (n)];
        u31 = u[(m - 2) * N + (n)];
        b31 = b[(m - 2) * N + (n)];
    }

    linear2(x - dx, s_4.x, x, s_1.x, x + dx, s_2.x, x - dx / 2.0, x + dx / 2.0, A, B);
    if (B <= 0)
    {
        s12.x = s_1.x;
    }
    else
    {
        s12.x = B;
    }
    if (A <= 0)
    {
        s14.x = s_1.x;
    }
    else
    {
        s14.x = A;
    }
    linear2(x - dx, s_4.y, x, s_1.y, x + dx, s_2.y, x - dx / 2.0, x + dx / 2.0, A, B);
    if ((B <= 0) || (grad_p == false))
    {
        s12.y = s_1.y;
    }
    else
    {
        s12.y = B;
    }
    if ((A <= 0) || (grad_p == false))
    {
        s14.y = s_1.y;
    }
    else
    {
        s14.y = A;
    }
    linear2(x - dx, u_4.x, x, u_1.x, x + dx, u_2.x, x - dx / 2.0, x + dx / 2.0, A, B);
    u12.x = B;
    u14.x = A;
    linear2(x - dx, u_4.y, x, u_1.y, x + dx, u_2.y, x - dx / 2.0, x + dx / 2.0, A, B);
    u12.y = B;
    u14.y = A;

    linear2(x - dx, b_4.x, x, b_1.x, x + dx, b_2.x, x - dx / 2.0, x + dx / 2.0, A, B);
    b12.x = B;
    b14.x = A;
    linear2(x - dx, b_4.y, x, b_1.y, x + dx, b_2.y, x - dx / 2.0, x + dx / 2.0, A, B);
    b12.y = B;
    b14.y = A;
    linear2(x - dx, b_4.z, x, b_1.z, x + dx, b_2.z, x - dx / 2.0, x + dx / 2.0, A, B);
    b12.z = B;
    b14.z = A;


    linear2(y - dy, s_3.x, y, s_1.x, y + dy, s_5.x, y - dy / 2.0, y + dy / 2.0, A, B);
    if (B <= 0)
    {
        s15.x = s_1.x;
    }
    else
    {
        s15.x = B;
    }
    if (A <= 0)
    {
        s13.x = s_1.x;
    }
    else
    {
        s13.x = A;
    }
    linear2(y - dy, s_3.y, y, s_1.y, y + dy, s_5.y, y - dy / 2.0, y + dy / 2.0, A, B);
    if ((B <= 0) || (grad_p == false))
    {
        s15.y = s_1.y;
    }
    else
    {
        s15.y = B;
    }
    if ((A <= 0) || (grad_p == false))
    {
        s13.y = s_1.y;
    }
    else
    {
        s13.y = A;
    }
    linear2(y - dy, u_3.x, y, u_1.x, y + dy, u_5.x, y - dy / 2.0, y + dy / 2.0, A, B);
    u15.x = B;
    u13.x = A;
    linear2(y - dy, u_3.y, y, u_1.y, y + dy, u_5.y, y - dy / 2.0, y + dy / 2.0, A, B);
    u15.y = B;
    u13.y = A;

    linear2(y - dy, b_3.x, y, b_1.x, y + dy, b_5.x, y - dy / 2.0, y + dy / 2.0, A, B);
    b15.x = B;
    b13.x = A;
    linear2(y - dy, b_3.y, y, b_1.y, y + dy, b_5.y, y - dy / 2.0, y + dy / 2.0, A, B);
    b15.y = B;
    b13.y = A;
    linear2(y - dy, b_3.z, y, b_1.z, y + dy, b_5.z, y - dy / 2.0, y + dy / 2.0, A, B);
    b15.z = B;
    b13.z = A;

    s21.x = linear(x, s_1.x, x + dx, s_2.x, x + 2.0 * dx, s21.x, x + dx / 2.0);
    if (s21.x <= 0) s21.x = s_2.x;
    s21.y = linear(x, s_1.y, x + dx, s_2.y, x + 2.0 * dx, s21.y, x + dx / 2.0);
    if ((s21.y <= 0) || (grad_p == false)) s21.y = s_2.y;
    u21.x = linear(x, u_1.x, x + dx, u_2.x, x + 2.0 * dx, u21.x, x + dx / 2.0);
    u21.y = linear(x, u_1.y, x + dx, u_2.y, x + 2.0 * dx, u21.y, x + dx / 2.0);
    b21.x = linear(x, b_1.x, x + dx, b_2.x, x + 2.0 * dx, b21.x, x + dx / 2.0);
    b21.y = linear(x, b_1.y, x + dx, b_2.y, x + 2.0 * dx, b21.y, x + dx / 2.0);
    b21.z = linear(x, b_1.z, x + dx, b_2.z, x + 2.0 * dx, b21.z, x + dx / 2.0);

    s41.x = linear(x, s_1.x, x - dx, s_4.x, x - 2.0 * dx, s41.x, x - dx / 2.0);
    if (s41.x <= 0) s41.x = s_4.x;
    s41.y = linear(x, s_1.y, x - dx, s_4.y, x - 2.0 * dx, s41.y, x - dx / 2.0);
    if ((s41.y <= 0) || (grad_p == false)) s41.y = s_4.y;
    u41.x = linear(x, u_1.x, x - dx, u_4.x, x - 2.0 * dx, u41.x, x - dx / 2.0);
    u41.y = linear(x, u_1.y, x - dx, u_4.y, x - 2.0 * dx, u41.y, x - dx / 2.0);
    b41.x = linear(x, b_1.x, x - dx, b_4.x, x - 2.0 * dx, b41.x, x - dx / 2.0);
    b41.y = linear(x, b_1.y, x - dx, b_4.y, x - 2.0 * dx, b41.y, x - dx / 2.0);
    b41.z = linear(x, b_1.z, x - dx, b_4.z, x - 2.0 * dx, b41.z, x - dx / 2.0);

    s31.x = linear(y, s_1.x, y - dy, s_3.x, y - 2.0 * dy, s31.x, y - dy / 2.0);
    if (s31.x <= 0) s31.x = s_3.x;
    s31.y = linear(y, s_1.y, y - dy, s_3.y, y - 2.0 * dy, s31.y, y - dy / 2.0);
    if ((s31.y <= 0) || (grad_p == false)) s31.y = s_3.y;
    u31.x = linear(y, u_1.x, y - dy, u_3.x, y - 2.0 * dy, u31.x, y - dy / 2.0);
    u31.y = linear(y, u_1.y, y - dy, u_3.y, y - 2.0 * dy, u31.y, y - dy / 2.0);
    b31.x = linear(y, b_1.x, y - dy, b_3.x, y - 2.0 * dy, b31.x, y - dy / 2.0);
    b31.y = linear(y, b_1.y, y - dy, b_3.y, y - 2.0 * dy, b31.y, y - dy / 2.0);
    b31.z = linear(y, b_1.z, y - dy, b_3.z, y - 2.0 * dy, b31.z, y - dy / 2.0);

    s51.x = linear(y, s_1.x, y + dy, s_5.x, y + 2.0 * dy, s51.x, y + dy / 2.0);
    if (s51.x <= 0) s51.x = s_5.x;
    s51.y = linear(y, s_1.y, y + dy, s_5.y, y + 2.0 * dy, s51.y, y + dy / 2.0);
    if ((s51.y <= 0) || (grad_p == false)) s51.y = s_5.y;
    u51.x = linear(y, u_1.x, y + dy, u_5.x, y + 2.0 * dy, u51.x, y + dy / 2.0);
    u51.y = linear(y, u_1.y, y + dy, u_5.y, y + 2.0 * dy, u51.y, y + dy / 2.0);
    b51.x = linear(y, b_1.x, y + dy, b_5.x, y + 2.0 * dy, b51.x, y + dy / 2.0);
    b51.y = linear(y, b_1.y, y + dy, b_5.y, y + 2.0 * dy, b51.y, y + dy / 2.0);
    b51.z = linear(y, b_1.z, y + dy, b_5.z, y + 2.0 * dy, b51.z, y + dy / 2.0);

    double Q = 1.0;
    double PQ = 0.0;
    double2 PS = { 0.0, 0.0 };
    double2 PU = { 0.0, 0.0 };
    double3 PB = { 0.0, 0.0, 0.0 };
    double Pdiv = 0.0;
    double P[8];
    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

    tmin = my_min(tmin, HLLDQ_Korolkov(s12.x, Q, s12.y, u12.x, u12.y, 0.0, b12.x, b12.y, b12.z, s21.x, Q, s21.y, //
        u21.x, u21.y, 0.0, b21.x, b21.y, b21.z, P, PQ, 1.0, 0.0, 0.0, dx, method));
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;
    Pdiv = Pdiv + dy * (b_1.x + b_2.x) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
    tmin = my_min(tmin, HLLDQ_Korolkov(s13.x, Q, s13.y, u13.x, u13.y, 0.0, b13.x, b13.y, b13.z, s31.x, Q, s31.y, //
        u31.x, u31.y, 0.0, b31.x, b31.y, b31.z, P, PQ, 0.0, -1.0, 0.0, dy, method));
    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;
    Pdiv = Pdiv - dx * (b_1.y + b_3.y) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
    tmin = my_min(tmin, HLLDQ_Korolkov(s14.x, Q, s14.y, u14.x, u14.y, 0.0, b14.x, b14.y, b14.z, s41.x, Q, s41.y, //
        u41.x, u41.y, 0.0, b41.x, b41.y, b41.z, P, PQ, -1.0, 0.0, 0.0, dx, method));
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;
    Pdiv = Pdiv - dy * (b_1.x + b_4.x) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
    tmin = my_min(tmin, HLLDQ_Korolkov(s15.x, Q, s15.y, u15.x, u15.y, 0.0, b15.x, b15.y, b15.z, s51.x, Q, s51.y, //
        u51.x, u51.y, 0.0, b51.x, b51.y, b51.z, P, PQ, 0.0, 1.0, 0.0, dy, method));
    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;
    Pdiv = Pdiv + dx * (b_1.y + b_5.y) / 2.0;


    if (*T > tmin)
    {
        // __threadfence();
        *T = tmin;
    }

    double dV = dx * dy;

    Pdiv = Pdiv + dV * b_1.y / y;
    //Pdiv = 0.0;


    s2[index].x = s_1.x - *T_do * PS.x / dV - *T_do * s_1.x * u_1.y / y;
    //s2[index].x = s_1.x - (*T_do / dV) * PS.x - *T_do * s_1.x * u_1.y / y;
    //s2[index].x = s_1.x - (*T_do / dV) * PS.x;   // � ���������� �����������
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, s2[index].x, *T_do, s_1.x);
        s2[index].x = s_1.x;
    }

    //u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * (PU.x + (b_1.x/cpi4)*Pdiv ) - *T_do * (s_1.x * u_1.y * u_1.x) / y) / s2[index].x;
    //u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * (PU.y + (b_1.y / cpi4) * Pdiv) - *T_do * s_1.x * u_1.y * u_1.y / y) / s2[index].x;
    //b2[index].x = (b_1.x - *T_do * (PB.x + u_1.x * Pdiv) / dV) - *T_do * (u_1.y*b_1.x - u_1.x * b_1.y)/y;
    //b2[index].y = (b_1.y - *T_do * (PB.y + u_1.y * Pdiv) / dV);
    //b2[index].z = (b_1.x - *T_do * (PB.z) / dV) - *T_do * (u_1.y * b_1.z) / y;

    //s2[index].y = ( ((s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * (PS.y + //
    //    (skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) / cpi4) * Pdiv) - //
    //    *T_do * u_1.y * (ggg * s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) / y) - //
    //    0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y) - kvv(b_1.x, b_1.y, b_1.z) / cpi8 ) * (ggg - 1);


    //s2[index].x = s_1.x - *T_do * PS.x / dV - *T_do * s_1.x * u_1.y / y;
    u2[index].x = (s_1.x * u_1.x - *T_do * (PU.x + (b_1.x / cpi4) * Pdiv) / dV - *T_do * (s_1.x * u_1.x * u_1.y - b_1.x * b_1.y / cpi4) / y) / s2[index].x;
    u2[index].y = (s_1.x * u_1.y - *T_do * (PU.y + (b_1.y / cpi4) * Pdiv) / dV - *T_do * (s_1.x * u_1.y * u_1.y + (kv(b_1.z) - kv(b_1.y)) / cpi4) / y) / s2[index].x;
    b2[index].x = (b_1.x - *T_do * (PB.x + u_1.x * Pdiv) / dV - *T_do * (u_1.y * b_1.x - b_1.y * u_1.x) / y);
    b2[index].y = (b_1.y - *T_do * (PB.y + u_1.y * Pdiv) / dV);
    b2[index].z = (b_1.z - *T_do * (PB.z) / dV);
    s2[index].y = (U8(s_1.x, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) - *T_do * (PS.y + (skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) / cpi4) * Pdiv)//
        / dV - *T_do * (((U8(s_1.x, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) + s_1.y + kvv(b_1.x, b_1.y, b_1.z) / cpi8) * u_1.y - b_1.y * skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) / cpi4) / y) //
        - 0.5 * s2[index].x * kvv(u2[index].x, u2[index].y, 0.0) - kvv(b2[index].x, b2[index].y, b2[index].z) / cpi8) * (ggg - 1.0);

    //� ���������� �������
    //u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * PU.x ) / s2[index].x;
    //u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * PU.y) / s2[index].x;

    //s2[index].y = ( ( (s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * PS.y ) - //
    //    0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);

    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }
}

__global__ void Kernel_GKR(double2* s, double2* u, double3* b, double2* s2, double2* u2, double3* b2, double* T, double* T_do, int i, int method)
{
    // ����� ������� ������� �� ������� � ������������. �������-������-��������
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // ���������� ������ ������� ������ (�������� ������)
    int n = index % N;                                   // ����� ������ �� x (�� 0)
    int m = (index - n) / N;                             // ����� ������ �� y (�� 0)
    double y = y_min + m * (y_max) / (M);
    double x = x_min + n * (x_max - x_min) / (N - 1);
    double dist = __dsqrt_rn(x * x + y * y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5, s_10, s_11, s_12, s_13, u_10, u_11, u_12, u_13;      // ���������� ���� ������� � ����� ������
    double3 b_1, b_2, b_3, b_4, b_5, b_10, b_11, b_12, b_13;
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // ������� �������
    double3 Pb12 = {0.0, 0.0, 0.0}, Pb13 = { 0.0, 0.0, 0.0 }, Pb14 = { 0.0, 0.0, 0.0 }, Pb15 = { 0.0, 0.0, 0.0 };
    double tmin = 1000;

    s_1 = s[index];
    u_1 = u[index];
    b_1 = b[index];


    /* double ss = 3.0;
     if (i > 20000)
     {
         ss = 2.0;
     }*/

    if ((dist < 2.0))  // Ƹ����� ��������� �������
    {
        // � ���� ������� �������� ���������� ������������� � �� �������� � �������� �������)
        s2[index] = s_1;
        u2[index] = u_1;
        b2[index] = b_1;
        return;
    }

    if ((m == M - 1))
    {
        s_5 = { 1.0, 1.0 / ggg };
        u_5 = { 0.0, 0.0 };
        /*if (u_5.y < 0)
        {
            u_5.y = 0.01;
        }*/
        b_5 = b_1;
    }
    else
    {
        s_5 = s[(m + 1) * N + n];
        u_5 = u[(m + 1) * N + n];
        b_5 = b[(m + 1) * N + n];
    }

    if ((n == N - 1))  // ������, �� ������� ������
    {
        s_2 = s_1;
        u_2 = u_1;
        if (u_2.x < 0)
        {
            u_2.x = 0.01;
        }
        b_2 = b_1;
    }
    else
    {
        s_2 = s[(m)*N + n + 1];
        u_2 = u[(m)*N + n + 1];
        b_2 = b[(m)*N + n + 1];
    }

    if (n == 0)
    {
        s_4 = s_1;
        u_4.y = u_1.y;
        u_4.x = -u_1.x;
        b_4.x = -b_1.x;
        b_4.y = -b_1.y;
        b_4.z = b_1.z;
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
        b_4 = b[(m)*N + n - 1];
    }

    if ((m == 0))
    {
        s_3 = s_1;
        u_3.x = u_1.x;
        u_3.y = -u_1.y;
        //b_3.x = -b_1.x;
        b_3.x = b_1.x;
        b_3.y = b_1.y;
        b_3.z = -b_1.z;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
        b_3 = b[(m - 1) * N + (n)];
    }

    double2 s_6, s_7, s_8, s_9;      // ���������� ���� ������� � ����� ������
    double2 u_6, u_7, u_8, u_9;      // ���������� ���� ������� � ����� ������
    double3 b_6, b_7, b_8, b_9;

    // ��������� ����� 10 - 13
    if (n == N - 1)
    {
        s_10 = s_5;
        u_10 = u_5;
        b_10 = b_5;
    }
    else if (m == M - 1)
    {
        s_10 = { 1.0, 1.0 / ggg };
        u_10 = { 0.0, 0.0 };
        b_10 = b_2;
    }
    else
    {
        s_10 = s[(m + 1) * N + n + 1];
        u_10 = u[(m + 1) * N + n + 1];
        b_10 = b[(m + 1) * N + n + 1];
    }
   

    if ((m == 0))
    {
        s_12 = s_4;
        u_12.x = u_4.x;
        u_12.y = -u_4.y;
        b_12.x = b_4.x;
        b_12.y = b_4.y;
        b_12.z = -b_4.z;

        s_11 = s_2;
        u_11.x = u_2.x;
        u_11.y = -u_2.y;
        b_11.x = b_2.x;
        b_11.y = b_2.y;
        b_11.z = -b_2.z;
    }
    else
    {
        if (n == 0)
        {
            s_12 = s_3;
            u_12.y = u_3.y;
            u_12.x = -u_3.x;
            b_12.x = -b_3.x;
            b_12.y = -b_3.y;
            b_12.z = b_3.z;
        }
        else
        {
            s_12 = s[(m - 1) * N + (n - 1)];
            u_12 = u[(m - 1) * N + (n - 1)];
            b_12 = b[(m - 1) * N + (n - 1)];
        }

        if (n == N - 1)
        {
            s_11 = s_3;
            u_11 = u_3;
            if (u_11.x < 0)
            {
                u_11.x = 0.01;
            }
            b_11 = b_3;
        }
        else
        {
            s_11 = s[(m - 1) * N + (n + 1)];
            u_11 = u[(m - 1) * N + (n + 1)];
            b_11 = b[(m - 1) * N + (n + 1)];
        }
        
    }

    if (n == 0)
    {
        s_13 = s_5;
        u_13.y = u_5.y;
        u_13.x = -u_5.x;
        b_13.x = -b_5.x;
        b_13.y = -b_5.y;
        b_13.z = b_5.z;
    }
    else
    {
        if (m == M - 1)
        {
            s_13 = { 1.0, 1.0 / ggg };
            u_13 = { 0.0, 0.0 };
            /*if (u_13.y < 0)
            {
                u_13.y = 0.01;
            }*/
            b_13 = b_4;
        }
        else
        {
            s_13 = s[(m + 1) * N + n - 1];
            u_13 = u[(m + 1) * N + n - 1];
            b_13 = b[(m + 1) * N + n - 1];
        }
    }

    double A = 0, B = 0;
    // ��������� ���������� �������-�������
    if (n >= N - 2)
    {
        s_6 = s_2;
        u_6 = u_2;
        if (u_6.x < 0)
        {
            u_6.x = 0.01;
        }
        b_6 = b_2;
    }
    else
    {
        s_6 = s[(m)*N + (n + 2)];
        u_6 = u[(m)*N + (n + 2)];
        b_6 = b[(m)*N + (n + 2)];
    }

    if (m >= M - 2) // ������ ������ ������� ��� ������
    {
        s_9 = s_5;
        u_9 = u_5;
        b_9 = b_5;
    }
    else
    {
        s_9 = s[(m + 2)*N + (n)];
        u_9 = u[(m + 2)*N + (n)];
        b_9 = b[(m + 2)*N + (n)];
    }


    if (n == 0)
    {
        s_8 = s_2;
        u_8.y = u_2.y;
        u_8.x = -u_2.x;
        b_8.x = -b_2.x;
        b_8.y = -b_2.y;
        b_8.z = b_2.z;
    }
    else if (n == 1)
    {
        s_8 = s_2;
        u_8.y = u_2.y;
        u_8.x = -u_2.x;
        b_8.x = -b_2.x;
        b_8.y = -b_2.y;
        b_8.z = b_2.z;
    }
    else
    {
        s_8 = s[(m) * N + (n - 2)];
        u_8 = u[(m) * N + (n - 2)];
        b_8 = b[(m) * N + (n - 2)];
    }


    if (m == 1)
    {
        s_7 = s_5;
        u_7.x = u_5.x;
        u_7.y = -u_5.y;
        b_7.x = b_5.x;
        b_7.y = b_5.y;
        b_7.z = -b_5.z;
    }
    else if (m == 0)
    {
        s_7 = s_3;
        u_7.x = u_3.x;
        u_7.y = -u_3.y;
        b_7.x = b_3.x;
        b_7.y = b_3.y;
        b_7.z = -b_3.z;
    }
    else
    {
        s_7 = s[(m - 2)*N + (n)];
        u_7 = u[(m - 2)*N + (n)];
        b_7 = b[(m - 2)*N + (n)];
    }

    // ��������� ���������

    double DX1[7], DY1[7], DX2[7], DY2[7], DX3[7], DY3[7], DX4[7], DY4[7], DX5[7], DY5[7];

    DX1[0] = dx * minmod( (s_2.x - s_1.x)/dx, (s_1.x - s_4.x) / dx);
    DX1[1] = dx * minmod((s_2.y - s_1.y) / dx, (s_1.y - s_4.y) / dx);
    DX1[2] = dx * minmod((u_2.x - u_1.x) / dx, (u_1.x - u_4.x) / dx);
    DX1[3] = dx * minmod((s_2.y - s_1.y) / dx, (s_1.y - s_4.y) / dx);
    DX1[4] = dx * minmod((b_2.x - b_1.x) / dx, (b_1.x - b_4.x) / dx);
    DX1[5] = dx * minmod((b_2.y - b_1.y) / dx, (b_1.y - b_4.y) / dx);
    DX1[6] = dx * minmod((b_2.z - b_1.z) / dx, (b_1.z - b_4.z) / dx);

    DX2[0] = dx * minmod((s_6.x - s_2.x) / dx, (s_2.x - s_1.x) / dx);
    DX2[1] = dx * minmod((s_6.y - s_2.y) / dx, (s_2.y - s_1.y) / dx);
    DX2[2] = dx * minmod((u_6.x - u_2.x) / dx, (u_2.x - u_1.x) / dx);
    DX2[3] = dx * minmod((s_6.y - s_2.y) / dx, (s_2.y - s_1.y) / dx);
    DX2[4] = dx * minmod((b_6.x - b_2.x) / dx, (b_2.x - b_1.x) / dx);
    DX2[5] = dx * minmod((b_6.y - b_2.y) / dx, (b_2.y - b_1.y) / dx);
    DX2[6] = dx * minmod((b_6.z - b_2.z) / dx, (b_2.z - b_1.z) / dx);

    DX4[0] = dx * minmod((s_1.x - s_4.x) / dx, (s_4.x - s_8.x) / dx);
    DX4[1] = dx * minmod((s_1.y - s_4.y) / dx, (s_4.y - s_8.y) / dx);
    DX4[2] = dx * minmod((u_1.x - u_4.x) / dx, (u_4.x - u_8.x) / dx);
    DX4[3] = dx * minmod((s_1.y - s_4.y) / dx, (s_4.y - s_8.y) / dx);
    DX4[4] = dx * minmod((b_1.x - b_4.x) / dx, (b_4.x - b_8.x) / dx);
    DX4[5] = dx * minmod((b_1.y - b_4.y) / dx, (b_4.y - b_8.y) / dx);
    DX4[6] = dx * minmod((b_1.z - b_4.z) / dx, (b_4.z - b_8.z) / dx);

    DX5[0] = dx * minmod((s_10.x - s_5.x) / dx, (s_5.x - s_13.x) / dx);
    DX5[1] = dx * minmod((s_10.y - s_5.y) / dx, (s_5.y - s_13.y) / dx);
    DX5[2] = dx * minmod((u_10.x - u_5.x) / dx, (u_5.x - u_13.x) / dx);
    DX5[3] = dx * minmod((s_10.y - s_5.y) / dx, (s_5.y - s_13.y) / dx);
    DX5[4] = dx * minmod((b_10.x - b_5.x) / dx, (b_5.x - b_13.x) / dx);
    DX5[5] = dx * minmod((b_10.y - b_5.y) / dx, (b_5.y - b_13.y) / dx);
    DX5[6] = dx * minmod((b_10.z - b_5.z) / dx, (b_5.z - b_13.z) / dx);

    DX3[0] = dx * minmod((s_11.x - s_3.x) / dx, (s_3.x - s_12.x) / dx);
    DX3[1] = dx * minmod((s_11.y - s_3.y) / dx, (s_3.y - s_12.y) / dx);
    DX3[2] = dx * minmod((u_11.x - u_3.x) / dx, (u_3.x - u_12.x) / dx);
    DX3[3] = dx * minmod((s_11.y - s_3.y) / dx, (s_3.y - s_12.y) / dx);
    DX3[4] = dx * minmod((b_11.x - b_3.x) / dx, (b_3.x - b_12.x) / dx);
    DX3[5] = dx * minmod((b_11.y - b_3.y) / dx, (b_3.y - b_12.y) / dx);
    DX3[6] = dx * minmod((b_11.z - b_3.z) / dx, (b_3.z - b_12.z) / dx);

    DY1[0] = dy * minmod((s_5.x - s_1.x) / dy, (s_1.x - s_3.x) / dy);
    DY1[1] = dy * minmod((s_5.y - s_1.y) / dy, (s_1.y - s_3.y) / dy);
    DY1[2] = dy * minmod((u_5.x - u_1.x) / dy, (u_1.x - u_3.x) / dy);
    DY1[3] = dy * minmod((s_5.y - s_1.y) / dy, (s_1.y - s_3.y) / dy);
    DY1[4] = dy * minmod((b_5.x - b_1.x) / dy, (b_1.x - b_3.x) / dy);
    DY1[5] = dy * minmod((b_5.y - b_1.y) / dy, (b_1.y - b_3.y) / dy);
    DY1[6] = dy * minmod((b_5.z - b_1.z) / dy, (b_1.z - b_3.z) / dy);
    
    DY5[0] = dy * minmod((s_9.x - s_5.x) / dy, (s_5.x - s_1.x) / dy);
    DY5[1] = dy * minmod((s_9.y - s_5.y) / dy, (s_5.y - s_1.y) / dy);
    DY5[2] = dy * minmod((u_9.x - u_5.x) / dy, (u_5.x - u_1.x) / dy);
    DY5[3] = dy * minmod((s_9.y - s_5.y) / dy, (s_5.y - s_1.y) / dy);
    DY5[4] = dy * minmod((b_9.x - b_5.x) / dy, (b_5.x - b_1.x) / dy);
    DY5[5] = dy * minmod((b_9.y - b_5.y) / dy, (b_5.y - b_1.y) / dy);
    DY5[6] = dy * minmod((b_9.z - b_5.z) / dy, (b_5.z - b_1.z) / dy);

    DY4[0] = dy * minmod((s_13.x - s_4.x) / dy, (s_4.x - s_12.x) / dy);
    DY4[1] = dy * minmod((s_13.y - s_4.y) / dy, (s_4.y - s_12.y) / dy);
    DY4[2] = dy * minmod((u_13.x - u_4.x) / dy, (u_4.x - u_12.x) / dy);
    DY4[3] = dy * minmod((s_13.y - s_4.y) / dy, (s_4.y - s_12.y) / dy);
    DY4[4] = dy * minmod((b_13.x - b_4.x) / dy, (b_4.x - b_12.x) / dy);
    DY4[5] = dy * minmod((b_13.y - b_4.y) / dy, (b_4.y - b_12.y) / dy);
    DY4[6] = dy * minmod((b_13.z - b_4.z) / dy, (b_4.z - b_12.z) / dy);

    DY2[0] = dy * minmod((s_10.x - s_2.x) / dy, (s_2.x - s_11.x) / dy);
    DY2[1] = dy * minmod((s_10.y - s_2.y) / dy, (s_2.y - s_11.y) / dy);
    DY2[2] = dy * minmod((u_10.x - u_2.x) / dy, (u_2.x - u_11.x) / dy);
    DY2[3] = dy * minmod((s_10.y - s_2.y) / dy, (s_2.y - s_11.y) / dy);
    DY2[4] = dy * minmod((b_10.x - b_2.x) / dy, (b_2.x - b_11.x) / dy);
    DY2[5] = dy * minmod((b_10.y - b_2.y) / dy, (b_2.y - b_11.y) / dy);
    DY2[6] = dy * minmod((b_10.z - b_2.z) / dy, (b_2.z - b_11.z) / dy);

    DY3[0] = dy * minmod((s_1.x - s_3.x) / dy, (s_3.x - s_7.x) / dy);
    DY3[1] = dy * minmod((s_1.y - s_3.y) / dy, (s_3.y - s_7.y) / dy);
    DY3[2] = dy * minmod((u_1.x - u_3.x) / dy, (u_3.x - u_7.x) / dy);
    DY3[3] = dy * minmod((s_1.y - s_3.y) / dy, (s_3.y - s_7.y) / dy);
    DY3[4] = dy * minmod((b_1.x - b_3.x) / dy, (b_3.x - b_7.x) / dy);
    DY3[5] = dy * minmod((b_1.y - b_3.y) / dy, (b_3.y - b_7.y) / dy);
    DY3[6] = dy * minmod((b_1.z - b_3.z) / dy, (b_3.z - b_7.z) / dy);

    // ������ ��������� ��������� ��� ���� ����� 1, 2, 3, 4, 5
    double Q1[7], Q2[7], Q3[7], Q4[7], Q5[7];
    Q1[0] = s_1.x;
    Q1[1] = s_1.y;
    Q1[2] = u_1.x;
    Q1[3] = u_1.y;
    Q1[4] = b_1.x;
    Q1[5] = b_1.y;
    Q1[6] = b_1.z;
    Q2[0] = s_2.x;
    Q2[1] = s_2.y;
    Q2[2] = u_2.x;
    Q2[3] = u_2.y;
    Q2[4] = b_2.x;
    Q2[5] = b_2.y;
    Q2[6] = b_2.z;
    Q3[0] = s_3.x;
    Q3[1] = s_3.y;
    Q3[2] = u_3.x;
    Q3[3] = u_3.y;
    Q3[4] = b_3.x;
    Q3[5] = b_3.y;
    Q3[6] = b_3.z;
    Q4[0] = s_4.x;
    Q4[1] = s_4.y;
    Q4[2] = u_4.x;
    Q4[3] = u_4.y;
    Q4[4] = b_4.x;
    Q4[5] = b_4.y;
    Q4[6] = b_4.z;
    Q5[0] = s_5.x;
    Q5[1] = s_5.y;
    Q5[2] = u_5.x;
    Q5[3] = u_5.y;
    Q5[4] = b_5.x;
    Q5[5] = b_5.y;
    Q5[6] = b_5.z;

    double QQ1[7], QQ2[7], QQ3[7], QQ4[7], QQ5[7];

    predictor(Q1, DX1, DY1, QQ1, *T_do/2.0, x, y);
    predictor(Q2, DX2, DY2, QQ2, *T_do / 2.0, x + dx, y);
    predictor(Q3, DX3, DY3, QQ3, *T_do / 2.0, x, y - dy);
    predictor(Q4, DX4, DY4, QQ4, *T_do / 2.0, x - dx, y);
    predictor(Q5, DX5, DY5, QQ5, *T_do / 2.0, x, y + dy);


    double Q = 1.0;
    double PQ = 0.0;
    double2 PS = { 0.0, 0.0 };
    double2 PU = { 0.0, 0.0 };
    double3 PB = { 0.0, 0.0, 0.0 };
    double Pdiv = 0.0;
    double P[8];
    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

    double ro_1, p_1, ro_2, p_2;

    ro_1 = (Q1[0] + QQ1[0]) / 2.0 + DX1[0] / 2.0;
    p_1 = (Q1[1] + QQ1[1]) / 2.0 + DX1[1] / 2.0;
    if (ro_1 <= 0.0)
    {
        ro_1 = (Q1[0] + QQ1[0]) / 2.0;
    }
    if (p_1 <= 0.0)
    {
        p_1 = (Q1[1] + QQ1[1]) / 2.0;
    }

    ro_2 = (Q2[0] + QQ2[0]) / 2.0 - DX2[0] / 2.0;
    p_2 = (Q2[1] + QQ2[1]) / 2.0 - DX2[1] / 2.0;
    if (ro_2 <= 0.0)
    {
        ro_2 = (Q2[0] + QQ2[0]) / 2.0;
    }
    if (p_2 <= 0.0)
    {
        p_2 = (Q2[1] + QQ2[1]) / 2.0;
    }

    tmin = my_min(tmin, HLLDQ_Korolkov(ro_1, Q, p_1, (Q1[2] + QQ1[2]) / 2.0 + DX1[2] / 2.0, (Q1[3] + QQ1[3]) / 2.0 + DX1[3] / 2.0, 0.0,//
        (Q1[4] + QQ1[4]) / 2.0 + DX1[4] / 2.0, (Q1[5] + QQ1[5]) / 2.0 + DX1[5] / 2.0, (Q1[6] + QQ1[6]) / 2.0 + DX1[6] / 2.0, //
        ro_2, Q, p_2, (Q2[2] + QQ2[2]) / 2.0 - DX2[2] / 2.0, (Q2[3] + QQ2[3]) / 2.0 - DX2[3] / 2.0, 0.0, //
        (Q2[4] + QQ2[4]) / 2.0 - DX2[4] / 2.0, (Q2[5] + QQ2[5]) / 2.0 - DX2[5] / 2.0, (Q2[6] + QQ2[6]) / 2.0 - DX2[6] / 2.0, P, PQ, 1.0, 0.0, 0.0, dx, 3));
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;
    Pdiv = Pdiv + dy * (b_1.x + b_2.x) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
    ro_1 = (Q1[0] + QQ1[0]) / 2.0 - DY1[0] / 2.0;
    p_1 = (Q1[1] + QQ1[1]) / 2.0 - DY1[1] / 2.0;
    if (ro_1 <= 0.0)
    {
        ro_1 = (Q1[0] + QQ1[0]) / 2.0;
    }
    if (p_1 <= 0.0)
    {
        p_1 = (Q1[1] + QQ1[1]) / 2.0;
    }

    ro_2 = (Q3[0] + QQ3[0]) / 2.0 + DY3[0] / 2.0;
    p_2 = (Q3[1] + QQ3[1]) / 2.0 + DY3[1] / 2.0;
    if (ro_2 <= 0.0)
    {
        ro_2 = (Q3[0] + QQ3[0]) / 2.0;
    }
    if (p_2 <= 0.0)
    {
        p_2 = (Q3[1] + QQ3[1]) / 2.0;
    }

    tmin = my_min(tmin, HLLDQ_Korolkov(ro_1, Q, p_1, (Q1[2] + QQ1[2]) / 2.0 - DY1[2] / 2.0, (Q1[3] + QQ1[3]) / 2.0 - DY1[3] / 2.0, 0.0,//
        (Q1[4] + QQ1[4]) / 2.0 - DY1[4] / 2.0, (Q1[5] + QQ1[5]) / 2.0 - DY1[5] / 2.0, (Q1[6] + QQ1[6]) / 2.0 - DY1[6] / 2.0, //
        ro_2, Q, p_2, (Q3[2] + QQ3[2]) / 2.0 + DY3[2] / 2.0, (Q3[3] + QQ3[3]) / 2.0 + DY3[3] / 2.0, 0.0, //
        (Q3[4] + QQ3[4]) / 2.0 + DY3[4] / 2.0, (Q3[5] + QQ3[5]) / 2.0 + DY3[5] / 2.0, (Q3[6] + QQ3[6]) / 2.0 + DY3[6] / 2.0, P, PQ, 0.0, -1.0, 0.0, dy, 3));

    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;
    Pdiv = Pdiv - dx * (b_1.y + b_3.y) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
    ro_1 = (Q1[0] + QQ1[0]) / 2.0 - DX1[0] / 2.0;
    p_1 = (Q1[1] + QQ1[1]) / 2.0 - DX1[1] / 2.0;
    if (ro_1 <= 0.0)
    {
        ro_1 = (Q1[0] + QQ1[0]) / 2.0;
    }
    if (p_1 <= 0.0)
    {
        p_1 = (Q1[1] + QQ1[1]) / 2.0;
    }

    ro_2 = (Q4[0] + QQ4[0]) / 2.0 + DX4[0] / 2.0;
    p_2 = (Q4[1] + QQ4[1]) / 2.0 + DX4[1] / 2.0;
    if (ro_2 <= 0.0)
    {
        ro_2 = (Q4[0] + QQ4[0]) / 2.0;
    }
    if (p_2 <= 0.0)
    {
        p_2 = (Q4[1] + QQ4[1]) / 2.0;
    }

    tmin = my_min(tmin, HLLDQ_Korolkov(ro_1, Q, p_1, (Q1[2] + QQ1[2]) / 2.0 - DX1[2] / 2.0, (Q1[3] + QQ1[3]) / 2.0 - DX1[3] / 2.0, 0.0,//
        (Q1[4] + QQ1[4]) / 2.0 - DX1[4] / 2.0, (Q1[5] + QQ1[5]) / 2.0 - DX1[5] / 2.0, (Q1[6] + QQ1[6]) / 2.0 - DX1[6] / 2.0, //
        ro_2, Q, p_2, (Q4[2] + QQ4[2]) / 2.0 + DX4[2] / 2.0, (Q4[3] + QQ4[3]) / 2.0 + DX4[3] / 2.0, 0.0, //
        (Q4[4] + QQ4[4]) / 2.0 + DX4[4] / 2.0, (Q4[5] + QQ4[5]) / 2.0 + DX4[5] / 2.0, (Q4[6] + QQ4[6]) / 2.0 + DX4[6] / 2.0, P, PQ, -1.0, 0.0, 0.0, dx, 3));

    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;
    Pdiv = Pdiv - dy * (b_1.x + b_4.x) / 2.0;

    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
    ro_1 = (Q1[0] + QQ1[0]) / 2.0 + DY1[0] / 2.0;
    p_1 = (Q1[1] + QQ1[1]) / 2.0 + DY1[1] / 2.0;
    if (ro_1 <= 0.0)
    {
        ro_1 = (Q1[0] + QQ1[0]) / 2.0;
    }
    if (p_1 <= 0.0)
    {
        p_1 = (Q1[1] + QQ1[1]) / 2.0;
    }

    ro_2 = (Q5[0] + QQ5[0]) / 2.0 - DY5[0] / 2.0;
    p_2 = (Q5[1] + QQ5[1]) / 2.0 - DY5[1] / 2.0;
    if (ro_2 <= 0.0)
    {
        ro_2 = (Q5[0] + QQ5[0]) / 2.0;
    }
    if (p_2 <= 0.0)
    {
        p_2 = (Q5[1] + QQ5[1]) / 2.0;
    }

    tmin = my_min(tmin, HLLDQ_Korolkov(ro_1, Q, p_1, (Q1[2] + QQ1[2]) / 2.0 + DY1[2] / 2.0, (Q1[3] + QQ1[3]) / 2.0 + DY1[3] / 2.0, 0.0,//
        (Q1[4] + QQ1[4]) / 2.0 + DY1[4] / 2.0, (Q1[5] + QQ1[5]) / 2.0 + DY1[5] / 2.0, (Q1[6] + QQ1[6]) / 2.0 + DY1[6] / 2.0, //
        ro_2, Q, p_2, (Q5[2] + QQ5[2]) / 2.0 - DY5[2] / 2.0, (Q5[3] + QQ5[3]) / 2.0 - DY5[3] / 2.0, 0.0, //
        (Q5[4] + QQ5[4]) / 2.0 - DY5[4] / 2.0, (Q5[5] + QQ5[5]) / 2.0 - DY5[5] / 2.0, (Q5[6] + QQ5[6]) / 2.0 - DY5[6] / 2.0, P, PQ, 0.0, 1.0, 0.0, dy, 3));

    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;
    Pdiv = Pdiv + dx * (b_1.y + b_5.y) / 2.0;


    if (*T > tmin)
    {
        // __threadfence();
        *T = tmin;
    }

    double dV = dx * dy;

    //Pdiv = Pdiv + dV * b_1.y / y;
    Pdiv = 0.0;


    s2[index].x = ((Q1[0] + QQ1[0])/2.0) - (*T_do/2.0) * PS.x / dV - (*T_do/2.0) * ((Q1[0] + QQ1[0])/2.0) * ((Q1[3] + QQ1[3])/2.0) / y;
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, s2[index].x, (*T_do/2.0), ((Q1[0] + QQ1[0])/2.0));
        s2[index].x = ((Q1[0] + QQ1[0])/2.0);
    }


    u2[index].x = (((Q1[0] + QQ1[0])/2.0) * ((Q1[2] + QQ1[2])/2.0) - (*T_do/2.0) * (PU.x + (((Q1[4] + QQ1[4])/2.0) / cpi4) * Pdiv) / dV - (*T_do/2.0) * (((Q1[0] + QQ1[0])/2.0) * ((Q1[2] + QQ1[2])/2.0) * ((Q1[3] + QQ1[3])/2.0) - ((Q1[4] + QQ1[4])/2.0) * ((Q1[5] + QQ1[5])/2.0) / cpi4) / y) / s2[index].x;
    u2[index].y = (((Q1[0] + QQ1[0])/2.0) * ((Q1[3] + QQ1[3])/2.0) - (*T_do/2.0) * (PU.y + (((Q1[5] + QQ1[5])/2.0) / cpi4) * Pdiv) / dV - (*T_do/2.0) * (((Q1[0] + QQ1[0])/2.0) * ((Q1[3] + QQ1[3])/2.0) * ((Q1[3] + QQ1[3])/2.0) + (kv(((Q1[6] + QQ1[6])/2.0)) - kv(((Q1[5] + QQ1[5])/2.0))) / cpi4) / y) / s2[index].x;
    b2[index].x = (((Q1[4] + QQ1[4])/2.0) - (*T_do/2.0) * (PB.x + ((Q1[2] + QQ1[2])/2.0) * Pdiv) / dV - (*T_do/2.0) * (((Q1[3] + QQ1[3])/2.0) * ((Q1[4] + QQ1[4])/2.0) - ((Q1[5] + QQ1[5])/2.0) * ((Q1[2] + QQ1[2])/2.0)) / y);
    b2[index].y = (((Q1[5] + QQ1[5])/2.0) - (*T_do/2.0) * (PB.y + ((Q1[3] + QQ1[3])/2.0) * Pdiv) / dV);
    b2[index].z = (((Q1[6] + QQ1[6])/2.0) - (*T_do/2.0) * (PB.z) / dV);
    s2[index].y = (U8(((Q1[0] + QQ1[0])/2.0), ((Q1[1] + QQ1[1])/2.0), ((Q1[2] + QQ1[2])/2.0), ((Q1[3] + QQ1[3])/2.0), 0.0, ((Q1[4] + QQ1[4])/2.0), ((Q1[5] + QQ1[5])/2.0), ((Q1[6] + QQ1[6])/2.0)) - (*T_do/2.0) * (PS.y + (skk(((Q1[2] + QQ1[2])/2.0), ((Q1[3] + QQ1[3])/2.0), 0.0, ((Q1[4] + QQ1[4])/2.0), ((Q1[5] + QQ1[5])/2.0), ((Q1[6] + QQ1[6])/2.0)) / cpi4) * Pdiv)//
        / dV - (*T_do/2.0) * (((U8(((Q1[0] + QQ1[0])/2.0), ((Q1[1] + QQ1[1])/2.0), ((Q1[2] + QQ1[2])/2.0), ((Q1[3] + QQ1[3])/2.0), 0.0, ((Q1[4] + QQ1[4])/2.0), ((Q1[5] + QQ1[5])/2.0), ((Q1[6] + QQ1[6])/2.0)) + ((Q1[1] + QQ1[1])/2.0) + kvv(((Q1[4] + QQ1[4])/2.0), ((Q1[5] + QQ1[5])/2.0), ((Q1[6] + QQ1[6])/2.0)) / cpi8) * ((Q1[3] + QQ1[3])/2.0) - ((Q1[5] + QQ1[5])/2.0) * skk(((Q1[2] + QQ1[2])/2.0), ((Q1[3] + QQ1[3])/2.0), 0.0, ((Q1[4] + QQ1[4])/2.0), ((Q1[5] + QQ1[5])/2.0), ((Q1[6] + QQ1[6])/2.0)) / cpi4) / y) //
        - 0.5 * s2[index].x * kvv(u2[index].x, u2[index].y, 0.0) - kvv(b2[index].x, b2[index].y, b2[index].z) / cpi8) * (ggg - 1.0);


    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }


}

__global__ void Kernel_TVD_old(double2* s, double2* u, double2* s2, double2* u2, double* T, double* T_do, int method)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // ���������� ������ ������� ������ (�������� ������)
    int n = index % N;                                   // ����� ������ �� x (�� 0)
    int m = (index - n) / N;                             // ����� ������ �� y (�� 0)
    double y = y_min + m * (y_max) / (M);
    double x = x_min + n * (x_max - x_min) / (N - 1);
    double dist = __dsqrt_rn(x * x + y * y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // ���������� ���� ������� � ����� ������
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // ������� �������
    double tmin = 1000;

    s_1 = s[index];
    u_1 = u[index];
    //if ((n == N - 1) || (m == M - 1) || (dist < 110)) // Ƹ����� ��������� �������
   if ((n == N - 1) || (dist < 110)) // Ƹ����� ��������� �������
    {
        // � ���� ������� �������� ���������� ������������� � �� �������� � �������� �������)
        s2[index] = s_1;
        u2[index] = u_1;
        return;
    }
    s_2 = s[(m)*N + n + 1];
    u_2 = u[(m)*N + n + 1];

    if (m == M - 1)
    {
        s_5 = s_1;
        u_5 = u_1;
    }
    else
    {
        s_5 = s[(m + 1) * N + n];
        u_5 = u[(m + 1) * N + n];
    }
    if ((n == 0))
    {
        s_4.x = s_1.x;
        s_4.y = 1.0 / (ggg * M_inf * M_inf);      // ������������ ��������
        u_4 = u_1;
        //if (u_1.x > 0.0)
        //{
        //    u_4.x = -0.3;              // ������� ������ ��������
        //}
        double Max = sqrt((u_1.x * u_1.x + u_1.y * u_1.y) / (ggg * s_1.y / s_1.x));
        if ( (u_1.x > -5.0)&&(Max < 1) )
        {
            u_4.x = -5.0;              // ������� �����- ������ ��������
        }
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
    }

    if ((m == 0))
    {
        s_3 = s_1;
        u_3.x = u_1.x;
        u_3.y = -u_1.y;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
    }

    double2 s12 = { 0.0 ,0.0 };
    double2 s13 = { 0.0 ,0.0 };
    double2 s14 = { 0.0 ,0.0 };
    double2 s15 = { 0.0 ,0.0 };
    double2 u12 = { 0.0 ,0.0 };
    double2 u13 = { 0.0 ,0.0 };
    double2 u14 = { 0.0 ,0.0 };
    double2 u15 = { 0.0 ,0.0 };
    double2 s21 = { 0.0 ,0.0 };
    double2 s31 = { 0.0 ,0.0 };
    double2 s41 = { 0.0 ,0.0 };
    double2 s51 = { 0.0 ,0.0 };
    double2 u21 = { 0.0 ,0.0 };
    double2 u31 = { 0.0 ,0.0 };
    double2 u41 = { 0.0 ,0.0 };
    double2 u51 = { 0.0 ,0.0 };
    double A = 0, B = 0;
    // ��������� ���������� �������-�������
    if (n > N - 3)
    {
        s21 = s_2;
        u21 = u_2;
    }
    else
    {
        s21 = s[(m) * N + (n + 2)];
        u21 = u[(m)*N + (n + 2)];
    }
    if (n == 0)
    {
        s41 = s_4;
        u41 = u_4;
    }
    else if (n == 1)
    {
        s41 = s_4;
        u41 = u_4;
        //if (u41.x > 0.0)
        //{
        //    u41.x = -0.3;             // ������� ������ ��������
        //}
        double Max = sqrt((u_4.x * u_4.x + u_4.y * u_4.y) / (ggg * s_4.y / s_4.x));
        if ((u41.x > -5.0) && (Max < 1))
        {
            u41.x = -5.0;              // ������� ������ ��������
        }
    }
    else
    {
        s41 = s[(m)*N + (n - 2)];
        u41 = u[(m)*N + (n - 2)];
    }
    if (m > M - 3)
    {
        s51 = s_5;
        u51 = u_5;
    }
    else
    {
        s51 = s[(m + 2)*N + (n)];
        u51 = u[(m + 2)*N + (n)];
    }
    if (m == 1)
    {
        s31 = s_3;
        u31.x = u_3.x;
        u31.y = -u_3.y;
    }
    else if (m == 0) 
    {
        s31 = s_5;
        u31.x = u_5.x;
        u31.y = -u_5.y;
    }
    else
    {
        s31 = s[(m - 2)*N + (n)];
        u31 = u[(m - 2)*N + (n)];
    }

    linear2(x - dx, s_4.x,      x, s_1.x,   x + dx, s_2.x,  x - dx/2.0, x + dx/2.0,     A, B);
    if (B <= 0)
    {
        s12.x = s_1.x;
    }
    else
    {
        s12.x = B;
    }
    if (A <= 0)
    {
        s14.x = s_1.x;
    }
    else
    {
        s14.x = A;
    }
    linear2(x - dx, s_4.y,       x, s_1.y,    x + dx, s_2.y,  x - dx / 2.0, x + dx / 2.0,    A, B);
    if ((B <= 0) || (grad_p == false) )
    {
        s12.y = s_1.y;
    }
    else
    {
        s12.y = B;
    }
    if ( (A <= 0) || (grad_p == false) )
    {
        s14.y = s_1.y;
    }
    else
    {
        s14.y = A;
    }
    linear2(x - dx, u_4.x,      x, u_1.x,   x + dx, u_2.x,       x - dx / 2.0, x + dx / 2.0,     A, B);
    u12.x = B;
    u14.x = A;
    linear2(x - dx, u_4.y,       x, u_1.y,   x + dx, u_2.y,     x - dx / 2.0, x + dx / 2.0,      A, B);
    u12.y = B;
    u14.y = A;

    linear2(y - dy, s_3.x,      y, s_1.x,       y + dy, s_5.x,      y - dy / 2.0, y + dy / 2.0,     A, B);
    if (B <= 0)
    {
        s15.x = s_1.x;
    }
    else
    {
        s15.x = B;
    }
    if (A <= 0)
    {
        s13.x = s_1.x;
    }
    else
    {
        s13.x = A;
    }
    linear2(y - dy, s_3.y,      y, s_1.y,       y + dy, s_5.y,      y - dy / 2.0, y + dy / 2.0,         A, B);
    if ((B <= 0) || (grad_p == false) )
    {
        s15.y = s_1.y;
    }
    else
    {
        s15.y = B;
    }
    if ( (A <= 0) || (grad_p == false) )
    {
        s13.y = s_1.y;
    }
    else
    {
        s13.y = A;
    }
    linear2(y - dy, u_3.x,      y, u_1.x,       y + dy, u_5.x,       y - dy / 2.0, y + dy / 2.0,        A, B);
    u15.x = B;
    u13.x = A;
    linear2(y - dy, u_3.y,       y, u_1.y,      y + dy, u_5.y,      y - dy / 2.0, y + dy / 2.0,         A, B);
    u15.y = B;
    u13.y = A;

    s21.x = linear(x, s_1.x,     x + dx, s_2.x,      x + 2.0 * dx, s21.x,       x + dx / 2.0);
    if (s21.x <= 0) s21.x = s_2.x;
    s21.y = linear(x, s_1.y,      x + dx, s_2.y,    x + 2.0 * dx, s21.y,    x + dx / 2.0);
    if ( (s21.y <= 0) || (grad_p == false) ) s21.y = s_2.y;
    u21.x = linear(x, u_1.x,    x + dx, u_2.x,      x + 2.0 * dx, u21.x,    x + dx / 2.0);
    u21.y = linear(x, u_1.y,    x + dx, u_2.y,      x + 2.0 * dx, u21.y,    x + dx / 2.0);

    s41.x = linear(x, s_1.x,    x - dx, s_4.x,      x - 2.0 * dx, s41.x,        x - dx / 2.0);
    if (s41.x <= 0) s41.x = s_4.x;
    s41.y = linear(x, s_1.y,    x - dx, s_4.y,      x - 2.0 * dx, s41.y,         x - dx / 2.0);
    if ((s41.y <= 0) || (grad_p == false) ) s41.y = s_4.y;
    u41.x = linear(x, u_1.x,    x - dx, u_4.x,      x - 2.0 * dx, u41.x,        x - dx / 2.0);
    u41.y = linear(x, u_1.y,    x - dx, u_4.y,      x - 2.0 * dx, u41.y,         x - dx / 2.0);

    s31.x = linear(y, s_1.x,        y - dy, s_3.x,      y - 2.0 * dy, s31.x,        y - dy / 2.0);
    if (s31.x <= 0) s31.x = s_3.x;
    s31.y = linear(y, s_1.y,        y - dy, s_3.y,      y - 2.0 * dy, s31.y,        y - dy / 2.0);
    if ( (s31.y <= 0) || (grad_p == false) ) s31.y = s_3.y;
    u31.x = linear(y, u_1.x,        y - dy, u_3.x,      y - 2.0 * dy, u31.x,        y - dy / 2.0);
    u31.y = linear(y, u_1.y,        y - dy, u_3.y,      y - 2.0 * dy, u31.y,        y - dy / 2.0);

    s51.x = linear(y, s_1.x,        y + dy, s_5.x,      y + 2.0 * dy, s51.x,        y + dy / 2.0);
    if (s51.x <= 0) s51.x = s_5.x;
    s51.y = linear(y, s_1.y,        y + dy, s_5.y,      y + 2.0 * dy, s51.y,        y + dy / 2.0);
    if ( (s51.y <= 0)||(grad_p == false) ) s51.y = s_5.y;
    u51.x = linear(y, u_1.x,        y + dy, u_5.x,      y + 2.0 * dy, u51.x,        y + dy / 2.0);
    u51.y = linear(y, u_1.y,        y + dy, u_5.y,      y + 2.0 * dy, u51.y,        y + dy / 2.0);


    if (method == 0)
    {
        tmin = min(tmin, HLL(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        tmin = min(tmin, HLL(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        tmin = min(tmin, HLL(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        tmin = min(tmin, HLL(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
    }
    else if (method == 1)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
        }
    }
    else if (method == 2)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
        }
    }
    else
    {
        printf("Error in method 2375\n");
    }


    if (*T > tmin)
    {
       // __threadfence();
        *T = tmin;
    }

    double2 PS = { 0,0 };
    double2 PU = { 0,0 };

    /*if ((x > 400) &&( x < 405) && (y < 5))
    {
        printf("%lf, %lf, %lf, %lf, %lf\n", Ps12.x, Ps13.x, Ps14.x, Ps15.x, Ps12.x* dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx);
    }*/

    PS.x = Ps12.x * dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx;
    PS.y = Ps12.y * dy + Ps13.y * dx + Ps14.y * dy + Ps15.y * dx;
    PU.x = Pu12.x * dy + Pu13.x * dx + Pu14.x * dy + Pu15.x * dx;
    PU.y = Pu12.y * dy + Pu13.y * dx + Pu14.y * dy + Pu15.y * dx;

    double dV = dx * dy;

    s2[index].x = s[index].x - (*T_do / dV) * PS.x - *T_do * s[index].x * u[index].y / y;
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf\n", x, y, s2[index].x);
        s2[index].x = 0.0001;
    }
    u2[index].x = (s[index].x * u[index].x - (*T_do / dV) * PU.x - *T_do * s[index].x * u[index].y * u[index].x / y) / s2[index].x;
    u2[index].y = (s[index].x * u[index].y - (*T_do / dV) * PU.y - *T_do * s[index].x * u[index].y * u[index].y / y) / s2[index].x;
    s2[index].y = (((s[index].y / (ggg - 1) + s[index].x * (u[index].x * u[index].x + u[index].y * u[index].y) * 0.5) - (*T_do / dV) * PS.y - //
        *T_do * u[index].y * (ggg * s[index].y / (ggg - 1) + s[index].x * (u[index].x * u[index].x + u[index].y * u[index].y) * 0.5) / y) - //
        0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);
    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }
}

__global__ void test(void)
{
    double2 s_1 = { 1, 0.0666666 };
    double2 u_1 = { -1, 0 };
    double2 s_2 = { 1, 0.0666666 };
    double2 u_2 = { -1, 0 };
    double2 P1, P2;
    Godunov_Solver_Alexashov(s_1, u_1, s_2, u_2, 1, 0, P1, P2, dy);
    printf("%lf\n", P1.x);
    Godunov_Solver_Alexashov(s_1, u_1, s_2, u_2, -1, 0, P1, P2, dy);
    printf("%lf\n", P1.x);
    
}

void print_file_mini(double2* host_s_p, double2* host_u_p, string name)
{
    ofstream fout;
    fout.open(name);
    int nn = (int)((N + Nmin - 1) / Nmin);
    int mm = (int)((M + Nmin - 1) / Nmin);
    fout << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"P\", \"Vx\", \"Vy\", \"Max\", \"T\", ZONE T = \"HP\", N = " << nn * mm //
        << " , E = " << (nn - 1) * (mm - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;
    for (int k = 0; k < K; k++)
    {
        int n = k % N;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / N;                             // ����� ������ �� y (�� 0)
        if ((n % Nmin != 0) || (m % Nmin != 0))
        {
            continue;
        }

        double y = y_min + m * (y_max - y_min) / (M - 1);
        double x = x_min + n * (x_max - x_min) / (N - 1);
        double Max = 0.0, Temp = 0.0;
        if (host_s_p[k].x > 0.0)
        {
            Max = sqrt((host_u_p[k].x * host_u_p[k].x + host_u_p[k].y * host_u_p[k].y) / (ggg * host_s_p[k].y / host_s_p[k].x));
            Temp = host_s_p[k].y / host_s_p[k].x;
        }
        fout << x << " " << y << " " << host_s_p[k].x << " " << host_s_p[k].y <<//
            " " << host_u_p[k].x << " " << host_u_p[k].y << " " << //
            Max << " " << Temp << endl;
    }

    for (int k = 0; k < nn * mm; k = k + 1)
    {
        int n = k % nn;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / nn;
        if ((m < mm - 1) && (n < nn - 1))
        {
            fout << m * nn + n + 1 << " " << m * nn + n + 2 << " " << (m + 1) * nn + n + 2 << " " << (m + 1) * nn + n + 1 << endl;
        }
    }
    fout.close();
}

void print_file_mini2(double2* host_s_p, double2* host_u_p, double3* host_b_p, string name)
{
    ofstream fout;
    fout.open(name);
    double r_o = 1.0; // 0.25320769;
    int nn = (int)((N + Nmin - 1) / Nmin);
    int mm = (int)((M + Nmin - 1) / Nmin);
    fout << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"P\", \"Vx\", \"Vy\", \"Bx\", \"By\", \"Bz\", \"Max\", \"T\", ZONE T = \"HP\", N = " << nn * mm //
        << " , E = " << (nn - 1) * (mm - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;
    for (int k = 0; k < K; k++)
    {
        int n = k % N;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / N;                             // ����� ������ �� y (�� 0)
        if ((n % Nmin != 0) || (m % Nmin != 0))
        {
            continue;
        }

        double y = y_min + m * (y_max - y_min) / (M - 1);
        double x = x_min + n * (x_max - x_min) / (N - 1);
        double Max = 0.0, Temp = 0.0;
        if (host_s_p[k].x > 0.0)
        {
            Max = sqrt((host_u_p[k].x * host_u_p[k].x + host_u_p[k].y * host_u_p[k].y) / (ggg * host_s_p[k].y / host_s_p[k].x));
            Temp = host_s_p[k].y / host_s_p[k].x;
        }
        fout << x * r_o << " " << y * r_o << " " << host_s_p[k].x << " " << host_s_p[k].y <<//
            " " << host_u_p[k].x << " " << host_u_p[k].y << " " << host_b_p[k].x << " " << host_b_p[k].y << " " << host_b_p[k].z << " " << //
            Max << " " << Temp << endl;
    }

    for (int k = 0; k < nn * mm; k = k + 1)
    {
        int n = k % nn;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / nn;
        if ((m < mm - 1) && (n < nn - 1))
        {
            fout << m * nn + n + 1 << " " << m * nn + n + 2 << " " << (m + 1) * nn + n + 2 << " " << (m + 1) * nn + n + 1 << endl;
        }
    }
    fout.close();
}

int main(void)
{
    double2* host_s, * host_u;
    double3* host_b;
    double2* s, * u;
    double3* b;
    double2* host_s2, * host_u2;
    double3* host_b2;
    int* host_i;
    double2* s2, * u2;
    double3* b2;
    int* dev_i;
    double* host_T, * host_T_do, * host_TT;
    double* T, * T_do, * TT;
    int size = K * sizeof(double2);
    int size2 = K * sizeof(double3);

    cudaEvent_t start, stop;
    cudaError_t cudaStatus;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //�������� ������ ��� device ����� ��� host_s, host_u
    cudaMalloc((void**)&s, size);
    cudaMalloc((void**)&u, size);
    cudaMalloc((void**)&b, size2);
    cudaMalloc((void**)&s2, size);
    cudaMalloc((void**)&u2, size);
    cudaMalloc((void**)&b2, size2);
    cudaMalloc((void**)&T, sizeof(double));
    cudaMalloc((void**)&T_do, sizeof(double));
    cudaMalloc((void**)&TT, sizeof(double));
    cudaMalloc((void**)&dev_i, sizeof(int));

    host_s = (double2*)malloc(size);
    host_u = (double2*)malloc(size);
    host_b = (double3*)malloc(size2);
    host_s2 = (double2*)malloc(size);
    host_u2 = (double2*)malloc(size);
    host_b2 = (double3*)malloc(size2);
    host_T = (double*)malloc(sizeof(double));
    host_T_do = (double*)malloc(sizeof(double));
    host_TT = (double*)malloc(sizeof(double));
    host_i = (int*)malloc(sizeof(int));

    *host_T = 10000000;
    *host_T_do = 0.00000001;
    *host_TT = 0.0;
    *host_i = 0;
    //cout << "dy" << dy << endl;
    //for (int k = 0; k < M; k++)  // ��������� ��������� �������
    //{
    //    double y = y_min + k * (y_max) / (M);
    //    cout << y << endl;
    //}
    
   /* double c1, c2, a1, a2, a3, a4, a5, a6, a7;
    ifstream fin;
    fin.open("instable_4.txt");

    for (int k = 0; k < K; k++)
    {
        fin >> c1 >> c2 >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7;
        host_s[k].x = a1;
        host_s[k].y = a2;
        host_u[k].x = a3;
        host_u[k].y = a4;
        host_b[k].x = a5;
        host_b[k].y = a6;
        host_b[k].z = a7;
        host_s2[k].x = a1;
        host_s2[k].y = a2;
        host_u2[k].x = a3;
        host_u2[k].y = a4;
        host_b2[k].x = a5;
        host_b2[k].y = a6;
        host_b2[k].z = a7;
    }
    fin.close();*/


    //double chi = 2.0;
    //double k_ = 0.1;
    //double l_ = 1.0;
    //double Max_ = 0.5;
    //double MM = kk_ / chi;
    double V_E = phi_0;
    double ro_E = 1.0/ (phi_0* phi_0 * rr_0 * rr_0); // MM / (4.0 * pi * V_E * rr_0 * rr_0);
    double P_E = ro_E * V_E * V_E / (ggg * M_0 * M_0);   // ��� ������ � ��������
    double B_E = sqrt(4.0 * pi) / (M_alf * rr_0);
    for (int k = 0; k < K; k++)  // ��������� ��������� �������
    {
        int n = k % N;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / N;                             // ����� ������ �� y (�� 0)
        double y = y_min + m * dy;
        double x = x_min + n * dx;

        //double dist2 = sqrt(kv(i->x + 0.8) + i->y * i->y + i->z * i->z);
        //double dist3 = kv(x) / kv(5.0) + kv(y) / kv(5.0);
        double dist = sqrt(x * x + y * y);

        if (dist < ddist * 1.01)
        {
            host_s[k] = { ro_E / pow(dist/ rr_0, 2.0), P_E * pow(rr_0 / dist, 2.0 * ggg) };
            host_u[k] = { V_E * x / dist, V_E * y / dist };
            host_s2[k] = host_s[k];
            host_u2[k] = host_u[k];

            double BE = B_E / (dist / rr_0);
            double the = acos(x / dist);
            double AA, BB, CC;

            dekard_skorost(y, 0.0, x, 0.0, BE * sin(the), 0.0, AA, BB, CC);
            host_b[k] = { 0.0, 0.0, BB };
            host_b2[k] = host_b[k];
        }
        else
        {
            host_s[k] = { 1.0, 1.0/ggg };
            host_u[k] = { 0.0, 0.0 };
            host_s2[k] = host_s[k];
            host_u2[k] = host_u[k];

            host_b[k] = { 0.0, 0.0, 0.0 };
            host_b2[k] = host_b[k];
        }
    }


    if (true)
    {
        double c1, c2, a1, a2, a3, a4, a5, a6, a7, a8;
        ifstream fin;
        fin.open("instable_cp14.txt");

        for (int k = 0; k < K; k++)
        {
            fin >> c1 >> c2 >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7;
            host_s[k].x = a1;
            host_s[k].y = a2;
            host_u[k].x = a3;
            host_u[k].y = a4;
            host_b[k].x = a5;
            host_b[k].y = a6;
            host_b[k].z = a7;

            host_s2[k].x = a1;
            host_s2[k].y = a2;
            host_u2[k].x = a3;
            host_u2[k].y = a4;
            host_b2[k].x = a5;
            host_b2[k].y = a6;
            host_b2[k].z = a7;
        }
        fin.close();

        for (int k = 0; k < K; k++)  // ��������� ��������� �������
        {
            int n = k % N;                                   // ����� ������ �� x (�� 0)
            int m = (k - n) / N;                             // ����� ������ �� y (�� 0)
            double y = y_min + m * dy;
            double x = x_min + n * dx;

            double dist = sqrt(x * x + y * y);

            if (dist < ddist * 1.001)
            {
                host_s[k] = { ro_E / pow(dist / rr_0, 2.0), P_E * pow(rr_0 / dist, 2.0 * ggg) };
                host_u[k] = { V_E * x / dist, V_E * y / dist };
                host_s2[k] = host_s[k];
                host_u2[k] = host_u[k];

                double BE = B_E / (dist / rr_0);
                double the = acos(x / dist);
                double AA, BB, CC;

                dekard_skorost(y, 0.0, x, 0.0, BE * sin(the), 0.0, AA, BB, CC);
                host_b[k] = { 0.0, 0.0, BB };
                host_b2[k] = host_b[k];
            }
        }
    }




    bool device = true;
    //�������� ���� �� device
    if (device)
    {
        cudaMemcpy(s, host_s, size, cudaMemcpyHostToDevice);
        cudaMemcpy(u, host_u, size, cudaMemcpyHostToDevice);
        cudaMemcpy(b, host_b, size2, cudaMemcpyHostToDevice);
        cudaMemcpy(s2, host_s2, size, cudaMemcpyHostToDevice);
        cudaMemcpy(u2, host_u2, size, cudaMemcpyHostToDevice);
        cudaMemcpy(b2, host_b2, size2, cudaMemcpyHostToDevice);
        cudaMemcpy(T, host_T, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(T_do, host_T_do, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(TT, host_TT, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_i, host_i, sizeof(int), cudaMemcpyHostToDevice);
    }


    //for (int i = 0; i < 30000; i = i + 2)  // ������� ����� �� ������� ������?
    //{
    //    // ��������� add() kernel �� GPU, ��������� ���������
    //    add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 0);
    //    funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
    //    add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 0);
    //    funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
    //}
    cout << "START" << endl;


    int meth = 2;  // HLL ������ ���! ����� �������

    // NO TVD
    for (int i = 0; i < 0; i = i + 2)  // ������� ����� �� ������� ������?
    {
        if (i % 5000 == 0)
        {
            cout << "Metod  " << meth << "   " << i <<  endl;
        }
        /*if (i > 100000)
        {
            meth = 3;
        }*/
        add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, b, s2, u2, b2, T, T_do, i, meth);
        //Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >  (s, u, b, s2, u2, b2, T, T_do, i, meth)
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        //Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, b2, s, u, b, T, T_do, i, meth)
        add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, b2, s, u, b, T, T_do, i, meth);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if ((i % 35000 == 0 && i > 2))
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("2000 step - Time:  %.2f sec\n", elapsedTime / 1000.0);
            cudaEventRecord(start, 0);
            cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_b, b, size2, cudaMemcpyDeviceToHost);
            string name = "02_12_" + to_string(i) + ".txt";
            print_file_mini2(host_s, host_u, host_b, name);
        }
    }

    // TVD
    for (int i = 0; i < 200000; i = i + 2)  // ������� ����� �� ������� ������?
    {
        if (i % 1000 == 0)
        {
            cout << "Metod  TVD  " << meth << "   " << i << endl;
        }
        /*if (i > 100000)
        {
            meth = 3;
        }*/
        //add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, b, s2, u2, b2, T, T_do, i, meth);
        Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, b, s2, u2, b2, T, T_do, i, meth);
            cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, b2, s, u, b, T, T_do, i, meth);
            //add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, b2, s, u, b, T, T_do, i, meth);
            cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if ((i % 45000 == 0 && i > 2))
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("2000 step - Time:  %.2f sec\n", elapsedTime / 1000.0);
            cudaEventRecord(start, 0);
            cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_b, b, size2, cudaMemcpyDeviceToHost);
            string name = "02_12_" + to_string(i) + ".txt";
            print_file_mini2(host_s, host_u, host_b, name);
        }
    }
    
  

    // copy device result back to host copy of c
    if (device)
    {
        cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_b, b, size2, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_s2, s2, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_u2, u2, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_b2, b2, size2, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_T, T, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_TT, TT, sizeof(double), cudaMemcpyDeviceToHost);


        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
    }

    printf("Time:  %.2f sec\n", elapsedTime/1000.0);

    if (device)
    {
        cudaFree(s);
        cudaFree(u);
        cudaFree(b);
        cudaFree(s2);
        cudaFree(u2);
        cudaFree(b2);
        cudaFree(T);
        cudaFree(T_do);
        cudaFree(TT);
        cudaFree(dev_i);
    }
    
    double r_o = 1.0; // 0.25320769;

    ofstream fout;
    fout.open("instable_cp19.txt");

    ofstream fout2;
    fout2.open("param_for_texplot.txt");

    ofstream fout5;
    fout5.open("param_for_texplot_mini.txt");

    ofstream fout3;
    fout3.open("param_y=0.txt");

    ofstream fout4;
    fout4.open("inform.txt");

    ofstream fout6;
    fout6.open("param_x=0.txt");

    fout2 << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"P\", \"Vx\", \"Vy\", \"Bx\", \"By\", \"Bz\", \"Max\", \"T\", ZONE T = \"HP\", N = " << K //
        << " , E = " << (N - 1) * (M - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;
    int nn = (int)((N + Nmin - 1) / Nmin);
    int mm = (int)((M + Nmin - 1) / Nmin);
    fout5 << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"P\", \"Vx\", \"Vy\", \"Bx\", \"By\", \"Bz\", \"Max\", \"T\", \"PP\", \"Fx\", \"Fy\", ZONE T = \"HP\", N = " << nn * mm //
        << " , E = " << (nn - 1)*(mm - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;


    fout3 << "TITLE = \"HP\"  VARIABLES = \"X\", \"Ro\", \"P\", \"Vx\", \"Bz\", \"Max\", \"T\", \"PP\", \"Fx\", \"Fy\", \"dP\", ZONE T = \"HP\"" << endl;
    fout6 << "TITLE = \"HP\"  VARIABLES = \"Y\", \"Ro\", \"P\", \"Vy\", \"Bz\", \"Max\", \"T\", \"PP\", \"Fx\", \"Fy\", \"dP\", ZONE T = \"HP\"" << endl;

    for (int k = 0; k < K; k++)
    {
        int n = k % N;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / N;                             // ����� ������ �� y (�� 0)
        double y = y_min + m * dy; // (y_max - y_min) / (M - 1);
        double x = x_min + n * dx; // (x_max - x_min) / (N - 1);
        fout << x << " " << y << " " << host_s[k].x << " " << host_s[k].y <<//
            " " << host_u[k].x << " " << host_u[k].y << " " << host_b[k].x << " " << host_b[k].y << " " << host_b[k].z  <<  endl;
    }

    //for (int k = 0; k < K; k++)
    //{
    //    int n = k % N;                                   // ����� ������ �� x (�� 0)
    //    int m = (k - n) / N;
    //    if ((m < M - 1) && (n < N - 1))
    //    {
    //        fout2 << m * N + n + 1 << " " << m * N + n + 2 << " " << (m + 1) * N + n + 2 << " " << (m + 1) * N + n + 1 << endl;
    //    }
    //}for (int k = 0; k < K; k++)
    //{
    //    int n = k % N;                                   // ����� ������ �� x (�� 0)
    //    int m = (k - n) / N;
    //    if ((m < M - 1) && (n < N - 1))
    //    {
    //        fout2 << m * N + n + 1 << " " << m * N + n + 2 << " " << (m + 1) * N + n + 2 << " " << (m + 1) * N + n + 1 << endl;
    //    }
    //}


    for (int k = 5; k < N - 1; k++)
    {
        int n = k % N;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / N;                             // ����� ������ �� y (�� 0)
        double y = y_min + m * dy; // (y_max - y_min) / (M - 1);
        double x = x_min + n * dx; // (x_max - x_min) / (N - 1);
        double zav = 0.0;
        int kk = (m + 1) * N + n;
        int kk2 = (m) * N + n + 1;
        zav = -host_b[k].z * (host_b[kk].z - host_b[k].z) / dy / (4.0 * pi);
        double Max = 0.0, Temp = 0.0;
        if (host_s[k].x > 0.0)
        {
            Max = sqrt((host_u[k].x * host_u[k].x + host_u[k].y * host_u[k].y) / (ggg * host_s[k].y / host_s[k].x));
            Temp = host_s[k].y / host_s[k].x;
        }

        int nn = (m)*N + n + 1;
        int nn2 = (m)*N + n - 1;

        double Fx = 0.0, Fy = 0.0;
        if (m < M - 1 && n < N - 1)
        {
            double dAr = 0.0, dAz = 0.0;

            if (n > 0)
            {
                dAz = (host_b[nn].z - host_b[nn2].z) / (2.0 * dx);
            }
            else
            {
                dAz = (host_b[nn].z - host_b[k].z) / (dx);
            }

            
            dAr = (host_b[kk].z - host_b[k].z) / (dy);

            Fx = host_b[k].z * dAz / (4.0 * pi);
            Fy = (-host_b[k].z * host_b[k].z / y - host_b[k].z * dAr) / (4.0 * pi);
        }

        fout3 << x << " " << host_s[k].x << " " << host_s[k].y <<//
            " " << host_u[k].x << " " << host_b[k].z << " " << Max << " " << Temp << " " << host_s[k].y + kv(host_b[k].z) / (8.0 * pi) << //
            " " << Fx << " " << Fy << " " << ((host_s[nn].y + kv(host_b[nn].z) / (8.0 * pi)) - (host_s[k].y + kv(host_b[k].z) / (8.0 * pi))) / dx << endl;
    }


    for (int m = 5; m < M - 1; m++)
    {
        int n = 0;
        double y = y_min + m * dy; // (y_max - y_min) / (M - 1);
        double x = x_min; // (x_max - x_min) / (N - 1);
        int kk = (m + 1) * N;
        int kk2 = (m - 1) * N;

        int nn = (m)*N + 0 + 1;
        int nn2 = (m)*N + 0 - 1;
        double zav = 0.0;
        int k = (m) * N + 0;
        zav = -host_b[k].z * (host_b[kk].z - host_b[k].z) / dy / (4.0 * pi);
        double Max = 0.0, Temp = 0.0;
        if (host_s[k].x > 0.0)
        {
            Max = sqrt((host_u[k].x * host_u[k].x + host_u[k].y * host_u[k].y) / (ggg * host_s[k].y / host_s[k].x));
            Temp = host_s[k].y / host_s[k].x;
        }

        double Fx = 0.0, Fy = 0.0;
        if (m < M - 1 && n < N - 1)
        {
            double dAr = 0.0, dAz = 0.0;

            if (n > 0)
            {
                dAz = (host_b[nn].z - host_b[nn2].z) / (2.0 * dx);
            }
            else
            {
                dAz = (host_b[nn].z - host_b[k].z) / (dx);
            }


            dAr = (host_b[kk].z - host_b[k].z) / (dy);

            Fx = host_b[k].z * dAz / (4.0 * pi);
            Fy = (-host_b[k].z * host_b[k].z / y - host_b[k].z * dAr) / (4.0 * pi);
        }

        fout6 << y << " " << host_s[k].x << " " << host_s[k].y <<//
            " " << host_u[k].y << " " << host_b[k].z << " " << Max << " " << Temp << " " << host_s[k].y + kv(host_b[k].z) / (8.0 * pi) << //
            " " << Fx << " " << Fy << " " << ((host_s[kk].y + kv(host_b[kk].z) / (8.0 * pi)) - (host_s[k].y + kv(host_b[k].z) / (8.0 * pi))) / dy << endl;
    }
    cout << "TT = " << *host_TT << endl;

    fout4 << "TT = " << *host_TT << "    N = " << N  << "   M = " << M << "   K = " << K  << endl;
    fout4 << "x_min = " << x_min << " " << "x_max = " << x_max << " " << "y_min = " << y_min << " " << "y_max = " << y_max << endl;
    fout4 << "M_inf = " << M_inf << " " << "phi_0 = " << phi_0 << "    M_A = " << M_alf << endl;

    int lll = 0;

  

    for (int k = 0; k < K; k++)
    {
        int n = k % N;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / N;                             // ����� ������ �� y (�� 0)

        double y = y_min + m * dy; // (y_max - y_min) / (M - 1);
        double x = x_min + n * dx; // (x_max - x_min) / (N - 1);

        if ((n % Nmin != 0) || (m % Nmin != 0))
        {
            continue;
        }
        lll++;

        int kk = (m + 1) * N + n;
        int kk2 = (m - 1) * N + n;

        int nn = (m) * N + n + 1;
        int nn2 = (m) * N + n - 1;

        double zav = 0.0;
        double Fx = 0.0, Fy = 0.0;
        if (m < M - 1 && n < N - 1)
        {
            double dAr = 0.0, dAz = 0.0;

            if (n > 0)
            {
                dAz = (host_b[nn].z - host_b[nn2].z) / (2.0 * dx);
            }
            else
            {
                dAz = (host_b[nn].z - host_b[k].z) / (dx);
            }

            if (m > 0)
            {
                dAr = (host_b[kk].z - host_b[kk2].z) / (2.0 * dy);
            }
            else
            {
                dAr = (host_b[kk].z - host_b[k].z) / (dy);
            }

            Fx = host_b[k].z * dAz / (4.0 * pi);
            Fy = (-host_b[k].z * host_b[k].z / y - host_b[k].z * dAr)/ (4.0 * pi);
        }

        
        double Max = 0.0, Temp = 0.0;
        if (host_s[k].x > 0.0)
        {
            Max = sqrt((host_u[k].x * host_u[k].x + host_u[k].y * host_u[k].y) / (ggg * host_s[k].y / host_s[k].x));
            Temp = host_s[k].y / host_s[k].x;
        }
        fout5 << x << " " << y << " " << host_s[k].x << " " << host_s[k].y <<//
            " " << host_u[k].x << " " << host_u[k].y << " "  << host_b[k].x << " " << host_b[k].y << " " << host_b[k].z << " " << //
            Max << " " << Temp << " "  << host_s[k].y + kv(host_b[k].z)/(8.0 * pi) << " " << Fx << " " << Fy << endl;
    }
    cout << lll << " = lll " << endl;
    cout << nn << " = nn " << endl;
    cout << mm << " = mm " << endl;

    for (int k = 0; k < nn * mm; k = k + 1)
    {
        int n = k % nn;                                   // ����� ������ �� x (�� 0)
        int m = (k - n) / nn;
        if ((m < mm - 1) && (n < nn - 1))
        {
            fout5 << m * nn + n + 1 << " " << m * nn + n + 2 << " " << (m + 1) * nn + n + 2 << " " << (m + 1) * nn + n + 1 << endl;
        }
    }

    fout.close();
    fout2.close();
    fout3.close();
    fout4.close();
    fout5.close();

    return 0;
}