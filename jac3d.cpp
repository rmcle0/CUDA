#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define ind(i, j, k) ((i) * L * L + (j) * L + (k))

#define L 500
#define ITMAX 100

int i, j, k, it;
double eps;
double MAXEPS = 0.5f;

int main(int argc, char **argv) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  double *A = new double[L * L * L];
  double *B = new double[L * L * L];

  double startt, endt;

  for (i = 0; i < L; i++)
    for (j = 0; j < L; j++)
      for (k = 0; k < L; k++) {
        A[ind(i, j, k)] = 0;
        if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 ||
            k == L - 1)
          B[ind(i, j, k)] = 0;
        else
          B[ind(i, j, k)] = 4 + i + j + k;
      }

  /* iteration loop */
  for (it = 1; it <= ITMAX; it++) {
    eps = 0;

    for (i = 1; i < L - 1; i++)
      for (j = 1; j < L - 1; j++)
        for (k = 1; k < L - 1; k++) {
          double tmp = std::abs(B[ind(i, j, k)] - A[ind(i, j, k)]);
          eps = Max(tmp, eps);
          A[ind(i, j, k)] = B[ind(i, j, k)];
        }

    for (i = 1; i < L - 1; i++)
      for (j = 1; j < L - 1; j++)
        for (k = 1; k < L - 1; k++)
          B[ind(i, j, k)] = (A[ind(i - 1, j, k)] + A[ind(i, j - 1, k)] +
                             A[ind(i, j, k - 1)] + A[ind(i, j, k + 1)] +
                             A[ind(i, j + 1, k)] + A[ind(i + 1, j, k)]) /
                            6.0f;

    printf(" IT = %4i   EPS = %14.7E\n", it, eps);
    if (eps < MAXEPS) break;
  }

  printf(" Jacobi3D Benchmark Completed.\n");
  printf(" Size            = %4d x %4d x %4d\n", L, L, L);
  printf(" Iterations      =       %12d\n", ITMAX);
  // TODO
  // printf(" Time in seconds =       %12.2lf\n", endt - startt);
  printf(" Operation type  =     floating point\n");
  // printf(" Verification    =       %12s\n", (fabs(eps - 5.058044) < 1e-11 ?
  // "SUCCESSFUL" : "UNSUCCESSFUL"));

  printf(" END OF Jacobi3D Benchmark\n");

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Time difference = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "[ms]" << std::endl;

  std::ofstream dump_file(std::string(argv[0]) + ".dump", std::ios::trunc);
  if (dump_file.is_open()) {
    for (i = 0; i < L; i++)
      for (j = 0; j < L; j++)
        for (k = 0; k < L; k++) dump_file << A[ind(i, j, k)];

    for (i = 0; i < L; i++)
      for (j = 0; j < L; j++)
        for (k = 0; k < L; k++) dump_file << B[ind(i, j, k)];
  }
  dump_file.close();

  return 0;
}
