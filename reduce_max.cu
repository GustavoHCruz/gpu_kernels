#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "chrono.c"

#define NUMBER_THREADS 1024

void reduce_thrust(float *max, float *h_vector, int n_elements)
{
  thrust::device_vector<float> vec_aux(h_vector, h_vector + n_elements);
  *max = thrust::reduce(thrust::device, vec_aux.begin(), vec_aux.end(), 0.0, thrust::maximum<float>());
}

__global__ void reduce_persistent(float *max, float *input, unsigned n_elements)
{
  __shared__ float shared_max[NUMBER_THREADS];

  unsigned int t_num = threadIdx.x;
  unsigned int t_index = blockIdx.x * blockDim.x + t_num;

  float local_max = -1;

  while (t_index < n_elements)
  {
    if (input[t_index] > local_max)
    {
      local_max = input[t_index];
    }
    t_index += gridDim.x * blockDim.x;
  }

  shared_max[t_num] = local_max;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (t_num < stride)
    {
      if (shared_max[t_num] < shared_max[t_num + stride])
      {
        shared_max[t_num] = shared_max[t_num + stride];
      }
    }
    __syncthreads();
  }

  if (t_num == 0)
  {
    atomicMax((int *)max, __float_as_int(shared_max[0]));
  }
}

__global__ void reduce_persistent_atomic(float *max, float *input, unsigned n_elements)
{
  __shared__ float shared_max;

  unsigned int t_num = threadIdx.x;
  unsigned int t_index = blockIdx.x * blockDim.x + t_num;

  if (t_num == 0)
  {
    shared_max = -1;
  }

  __syncthreads();

  while (t_index < n_elements)
  {
    atomicMax((int *)&shared_max, __float_as_int(input[t_index]));
    t_index += gridDim.x * blockDim.x;
  }
  __syncthreads();

  if (t_num == 0)
  {
    atomicMax((int *)max, __float_as_int(shared_max));
  }
}

void initialize_input_vector(float *input, int n_elements)
{
  for (int i = 0; i < n_elements; i++)
  {
    int a = rand();
    int b = rand();

    float v = a * 100.0 + b;

    input[i] = v;
  }
}

void print_greetings(unsigned n_elements, unsigned n_repetitions)
{
  printf("\n\n======================================================================\n");
  printf("Starting the Reduce Max procedure (parallel).\n");
  printf("Running The Kernels 'Thrust', 'Persistent' and 'Pesistent Atomic' for:\n");
  printf("- Number of Elements: %i\n", n_elements);
  printf("- Number of Repetitions: %i\n", n_repetitions);
}

void print_kernel_results(int kernel_number, float awnser, unsigned n_elements, unsigned n_repetitions, chronometer_t chrono, double *total_time)
{
  double total_chrono = (double)chrono_gettotal(&chrono);
  *total_time = total_chrono / ((double)1000 * 1000 * 1000);
  double ops = n_elements / *total_time;

  double mflops = ops / 1e6;

  printf("\n========== Report For ");
  if (kernel_number == 0)
    printf("Thrust Version ==========\n");
  else if (kernel_number == 1)
    printf("Persistent Version ==========\n");
  else
    printf("Persistent Atomic Version ==========\n");

  printf("Max Value Reduced (awnser): %.1f\n", awnser);
  printf("Total Time (seconds): %lf\n", *total_time);
  printf("Throughput: %.2lf MFLOPS\n", mflops);
  printf("==================================================\n");
}

void print_results(double thrust_time, double persistent_time, double persistent_atomic_time)
{
  double persistent = thrust_time / persistent_time;
  double persistent_atomic = thrust_time / persistent_atomic_time;

  printf("\nTime Comparasion beetween the Kernels:\n");
  if (thrust_time > persistent_time)
  {
    printf("- The Persistent Version Performs %lfx times Faster Than Thrust.\n", persistent);
  }
  else
  {
    printf("- The Persistent Version Performs at %lf speed of Thrust.\n", persistent);
  }

  if (thrust_time > persistent_atomic_time)
  {
    printf("- The Persistent Atomic Version Performs %lfx times Faster Than Thrust.\n", persistent_atomic);
  }
  else
  {
    printf("- The Persistent Atomic Version Performs at %lf speed of Thrust.\n", persistent_atomic);
  }
  printf("======================================================================\n\n");
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Correct usage: ./reduce <number_elements> <number_repetitions>\n");
    return -1;
  }

  chronometer_t chrono;

  unsigned n_elements = std::atoi(argv[1]);
  unsigned short n_repetitions = std::atoi(argv[2]);

  print_greetings(n_elements, n_repetitions);

  unsigned input_total_size = n_elements * sizeof(float);

  float *h_input = (float *)malloc(input_total_size), h_awnser;
  float *d_input, *d_awnser;
  cudaMalloc(&d_input, input_total_size);
  cudaMalloc(&d_awnser, sizeof(float));

  initialize_input_vector(h_input, n_elements);

  cudaMemcpy(d_input, h_input, input_total_size, cudaMemcpyHostToDevice);

  int number_blocks = (n_elements + NUMBER_THREADS - 1) / NUMBER_THREADS;

  double thrust_time, persistent_time, persistent_atomic_time;

  chrono_reset(&chrono);
  chrono_start(&chrono);
  for (int i = 0; i < n_repetitions; i++)
  {
    reduce_thrust(&h_awnser, h_input, n_elements);
  }
  cudaDeviceSynchronize();
  chrono_stop(&chrono);
  print_kernel_results(0, h_awnser, n_elements, n_repetitions, chrono, &thrust_time);

  chrono_reset(&chrono);
  chrono_start(&chrono);
  for (int i = 0; i < n_repetitions; i++)
  {
    cudaMemcpy(d_awnser, &h_awnser, sizeof(float), cudaMemcpyHostToDevice);
    reduce_persistent<<<number_blocks, NUMBER_THREADS>>>(d_awnser, d_input, n_elements);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_awnser, d_awnser, sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  chrono_stop(&chrono);
  print_kernel_results(1, h_awnser, n_elements, n_repetitions, chrono, &persistent_time);

  chrono_reset(&chrono);
  chrono_start(&chrono);
  for (int i = 0; i < n_repetitions; i++)
  {
    cudaMemcpy(d_awnser, &h_awnser, sizeof(float), cudaMemcpyHostToDevice);
    reduce_persistent_atomic<<<number_blocks, NUMBER_THREADS>>>(d_awnser, d_input, n_elements);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_awnser, d_awnser, sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  chrono_stop(&chrono);
  print_kernel_results(2, h_awnser, n_elements, n_repetitions, chrono, &persistent_atomic_time);

  free(h_input);
  cudaFree(d_input);
  cudaFree(d_awnser);

  print_results(thrust_time, persistent_time, persistent_atomic_time);

  return 0;
}