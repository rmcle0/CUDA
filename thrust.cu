#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace std;

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define L 500
#define ITMAX 100
#define ind(i, j, k) ((i) * L * L + (j) * L + (k))

__global__ void B_eq_sum_A(double *A, double *B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < L - 1 && j < L - 1) {
    for (int k = 1; k < L - 1; k++) {
      B[ind(i, j, k)] =
          (A[ind(i - 1, j, k)] + A[ind(i, j - 1, k)] + A[ind(i, j, k - 1)] +
           A[ind(i, j, k + 1)] + A[ind(i, j + 1, k)] + A[ind(i + 1, j, k)]) /
          6.0;
    }
  }
}

template <typename T>
struct absolute_value : public unary_function<T, T> {
  __host__ __device__ T operator()(const T &x) const {
    return x < T(0) ? -x : x;
  }
};

template <typename T>
struct zero : public unary_function<T, T> {
  __host__ __device__ T operator()(const T &x) const { return 0; }
};

int i, j, k, it;
double MAXEPS = 0.5f;

int main(int argc, char **argv) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  thrust::device_vector<double> A_device(L * L * L), B_device(L * L * L),
      A_B_device(L * L * L);
  thrust::host_vector<double> A(L * L * L), B(L * L * L);

  // avoid writing to cuda device for one copying at a time
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

  thrust::copy(A.begin(), A.end(), A_device.begin());
  thrust::copy(B.begin(), B.end(), B_device.begin());

  /* iteration loop */
  for (it = 1; it <= ITMAX; it++) {
    thrust::transform(A_device.begin(), A_device.end(), B_device.begin(),
                      A_B_device.begin(), thrust::minus<double>());
    double eps_device = thrust::transform_reduce(
        A_B_device.begin(), A_B_device.end(), absolute_value<double>(), 0.0,
        thrust::maximum<double>());

    thrust::copy(B_device.begin(), B_device.end(), A_device.begin());

    double *A_device_ptr = thrust::raw_pointer_cast(A_device.data());
    double *B_device_ptr = thrust::raw_pointer_cast(B_device.data());

    B_eq_sum_A<<<dim3(L / 32 + 1, L / 32 + 1, 1), dim3(32, 32, 1)>>>(
        A_device_ptr, B_device_ptr);

    printf(" IT = %4i   EPS = %14.7E\n", it, eps_device);
    if (eps_device < MAXEPS) break;
  }

  printf(" Jacobi3D Benchmark Completed.\n");
  printf(" Size            = %4d x %4d x %4d\n", L, L, L);
  printf(" Iterations      =       %12d\n", ITMAX);
  // TODO
  // printf(" Time in seconds =       %12.2lf\n",endt-startt);
  printf(" Operation type  =     floating point\n");
  // printf(" Verification    =       %12s\n", (fabs(eps-5.058044) < 1e-11 ?
  // "SUCCESSFUL" : "UNSUCCESSFUL"));

  printf(" END OF Jacobi3D Benchmark\n");

  A = A_device;
  B = B_device;

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
