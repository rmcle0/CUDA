#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace std;

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define ind(i, j, k) ((i) * L * L + (j) * L + (k))

__global__ void B_eq_sum_A(double *A, double *B, int L) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (k > 0 && j > 0 && k < L - 1 && j < L - 1) {
    for (int i = 1; i < L - 1; i++) {
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
  if(argc < 4){
    std::cout<<"./exec file_output|false 500 100" << std::endl;
    return 1;
  }
  bool file_output =  std::string(argv[1]) == std::string("file_output");
  int L = std::stoi(argv[2]);
  int ITMAX = std::stoi(argv[3]);



  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t1, t2;


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

    t1 = std::chrono::steady_clock::now();

    thrust::transform(A_device.begin(), A_device.end(), B_device.begin(),
                      A_B_device.begin(), thrust::minus<double>());

    t2 = std::chrono::steady_clock::now();
    std::cout << "1) Time difference = "  << (std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() ) << "[ms]" << std::endl;


    double eps_device = thrust::transform_reduce(
        A_B_device.begin(), A_B_device.end(), absolute_value<double>(), 0.0,
        thrust::maximum<double>());

    double *A_device_ptr = thrust::raw_pointer_cast(A_device.data());
    double *B_device_ptr = thrust::raw_pointer_cast(B_device.data());

    t1 = std::chrono::steady_clock::now();

    if(it % 2 == 1)
      B_eq_sum_A<<<dim3(L / 32 + 1, L / 32 + 1, 1), dim3(32, 32, 1)>>>(  B_device_ptr, A_device_ptr, L);
    else 
      B_eq_sum_A<<<dim3(L / 32 + 1, L / 32 + 1, 1), dim3(32, 32, 1)>>>(  A_device_ptr, B_device_ptr, L);

    cudaStreamSynchronize(0);

    t2 = std::chrono::steady_clock::now();
    std::cout << "2) Time difference = "  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "[ms]" << std::endl;



    printf(" IT = %4i   EPS = %14.7E\n", it, eps_device);
    if (eps_device < MAXEPS) break;
  }


  printf(" Jacobi3D Benchmark Completed.\n");
  printf(" Size            = %4d x %4d x %4d\n", L, L, L);
  printf(" Iterations      =       %12d\n", ITMAX);
  printf(" Operation type  =     floating point\n");
  printf(" END OF Jacobi3D Benchmark\n");

  A = A_device;
  B = B_device;

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::cout << "Size = " << L << std::endl;
  std::cout << "Time difference = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "[ms]" << std::endl;



  if(file_output){
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
  }

  return 0;
}
