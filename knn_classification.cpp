#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <iostream>
#include <limits>

using namespace cl::sycl;

#define BLOCK_SIZE 256

int mode_fn(int a[], int n) {
    int b[n];
    
    int max = *std::max_element(a, a+n);
    int t = max + 1;
    int count[t];
    
    for (int i = 0; i < t; i++)
        count[i] = 0;
    for (int i=0; i < n; i++)
        count[a[i]]++;
    
    int mode = 0;
    int k = count[0];
    for (int i = 1; i < t; i++) {
        if (count[i] > k) {
            k = count[i];
            mode = i;
        }
    }
    return mode;
}

void bs (const int seq_len, const int two_power, int *a, int *b, int *c, sycl::nd_item<3> item_ct1) {

  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);

  int seq_num = i / seq_len;
  int swapped_ele = -1;
  int h_len = seq_len / 2;

  if (i < (seq_len * seq_num) + h_len) swapped_ele = i + h_len;

  int odd = seq_num / two_power;
  bool increasing = ((odd % 2) == 0);

  // Swap the elements in the bitonic sequence if needed
  if (swapped_ele != -1) {
    if (((c[i] > c[swapped_ele]) && increasing) ||
    ((c[i] < c[swapped_ele]) && !increasing)) {
      int temp_a = a[i];
      int temp_b = b[i];
      int temp_c = c[i];
      a[i] = a[swapped_ele];
      b[i] = b[swapped_ele];
      c[i] = c[swapped_ele];
      a[swapped_ele] = temp_a;
      b[swapped_ele] = temp_b;
      c[swapped_ele] = temp_c;
    }
  }
}

void ParallelBitonicSort(int data[], int label[], int distance[], int n) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  time_t start, end;
    
  std::cout << "device: " << q_ct1.get_device().get_info <info::device::name>() << std::endl; 

  // n: the exponent used to set the array size. Array size = power(2, n)
  int size = sycl::pown((float)(2), n);

  // SYCL buffer allocated for device 
  int *a, *b, *c;
  a = sycl::malloc_device<int>(size, q_ct1);
  b = sycl::malloc_device<int>(size, q_ct1);
  c = sycl::malloc_device<int>(size, q_ct1);
  q_ct1.memcpy(a, data, sizeof(int) * size).wait();
  q_ct1.memcpy(b, label, sizeof(int) * size).wait();
  q_ct1.memcpy(c, distance, sizeof(int) * size).wait();

  time(&start);
  // step from 0, 1, 2, ...., n-1
  for (int step = 0; step < n; step++) {
    // for each step s, stage goes s, s-1, ..., 0
    for (int stage = step; stage >= 0; stage--) {
      int seq_len = sycl::pown((float)(2), stage + 1);
#if DEBUG
      int num_seq = pow(2, (n - stage - 1));  // Used for debug purpose.
      std::cout << "step num:" << step << " stage num:" << stage
                << " num_seq:" << num_seq << "(" << seq_len << ") => " << std::endl;
#endif
      // Constant used in the kernel: 2**(step-stage).
      int two_power = 1 << (step - stage);        
      q_ct1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, size / BLOCK_SIZE) *
                                  sycl::range<3>(1, 1, BLOCK_SIZE),
                              sycl::range<3>(1, 1, BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {           
                bs(seq_len, two_power, a, b, c, item_ct1);
            });
      });
      dev_ct1.queues_wait_and_throw();
    }  // end stage
  }    // end step
    
  time(&end);
    
  std::cout << "Parallel Bitonic Sort | start: " << start << ", end: " << end << ", duration: " << end - start << " seconds " << std::endl;
  q_ct1.memcpy(data, a, sizeof(int) * size).wait();
  q_ct1.memcpy(label, b, sizeof(int) * size).wait();
  q_ct1.memcpy(distance, c, sizeof(int) * size).wait();
  sycl::free(a, q_ct1);
  sycl::free(b, q_ct1);
  sycl::free(c, q_ct1);
}

// Loop over the bitonic sequences at each stage in serial.
void SwapElements(int step, int stage, int num_sequence, int seq_len,
                  int *data_array, int *label_array, int *distance_array) {
  for (int seq_num = 0; seq_num < num_sequence; seq_num++) {
    int odd = seq_num / (sycl::pown((float)(2), (step - stage)));
    bool increasing = ((odd % 2) == 0);

    int h_len = seq_len / 2;

    // For all elements in a bitonic sequence, swap them if needed
    for (int i = seq_num * seq_len; i < seq_num * seq_len + h_len; i++) {
      int swapped_ele = i + h_len;

      if (((distance_array[i] > distance_array[swapped_ele]) && increasing) ||
          ((distance_array[i] < distance_array[swapped_ele]) && !increasing)) {
        int temp_a = data_array[i];
        int temp_b = label_array[i];
        int temp_c = distance_array[i];
        data_array[i] = data_array[swapped_ele];
        label_array[i] = label_array[swapped_ele];
        distance_array[i] = distance_array[swapped_ele];
        data_array[swapped_ele] = temp_a;
        label_array[swapped_ele] = temp_b;
        distance_array[swapped_ele] = temp_c;
      }
    }  // end for all elements in a sequence
  }    // end all sequences
}

inline void BitonicSort(int a[], int b[], int c[], int n) {
  // n: the exponent indicating the array size = 2 ** n.

  time_t start, end;
    
  time(&start);
  // step from 0, 1, 2, ...., n-1
  for (int step = 0; step < n; step++) {
    // for each step s, stage goes s, s-1,..., 0
    for (int stage = step; stage >= 0; stage--) {
      // Sequences (same size) are formed at each stage.
      int num_sequence = sycl::pown((float)(2), (n - stage - 1));
      // The length of the sequences (2, 4, ...).
      int sequence_len = sycl::pown((float)(2), stage + 1);

      SwapElements(step, stage, num_sequence, sequence_len, a, b, c);
    }
  }
    
  time(&end);
  std::cout << "bitonic sort | start: " << start << ", end: " << end << ", duration: " << end - start << " seconds " << std::endl;
}

// Function showing the array.
void DisplayArray(int a[], int array_size) {
  for (int i = 0; i < array_size; ++i) {
      std::cout << a[i] << " ";
  }
  std::cout << "\n";
}

void Usage(std::string prog_name, int exponent) {
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: " << prog_name << " n k \n\n";
  std::cout << " n: Integer exponent presenting the size of the input array. "
               "The number of element in\n";
  std::cout << "    the array must be power of 2 (e.g., 1, 2, 4, ...). Please "
               "enter the corresponding\n";
  std::cout << "    exponent between 0 and " << exponent - 1 << ".\n";
  std::cout << " k: Seed used to generate a random sequence.\n";
}

int main(int argc, char *argv[]) {
    
    int n, seed, size, k=5;
    int exp_max = log2(std::numeric_limits<int>::max());
    time_t start, end;
    
    try {
        n = std::stoi(argv[1]);

        if (n < 0 || n >= exp_max) {
          Usage(argv[0], exp_max);
          return -1;
        }
        
        seed = std::stoi(argv[2]);
        size = sycl::pown((float)(2), n);
    } catch (...) {
        Usage(argv[0], exp_max);
        return -1;
    }

  std::cout << "\nArray size: " << size << ", seed: " << seed << "\n";

  // Memory allocated
  int *data = (int *)malloc(size * sizeof(int));
  int *label = (int *)malloc(size * sizeof(int));
  int *distance = (int *)malloc(size * sizeof(int));

  // Initialize the array randomly using a seed.
  srand(seed);
    
  int query = rand() % 50 + 1;
  std::cout << "query: " << query << std::endl;

  for (int i = 0; i < size; i++) {
    data[i] = rand() % 50 + 1;
    label[i] = rand() % 2;
    distance[i] = sqrt(pow(data[i] - query, 2));
  }
    
// #if DEBUG
  std::cout << "\ndata before:\n";
  DisplayArray(data, k);
  DisplayArray(label, k);
  std::cout << "\ndistance before:\n";
  DisplayArray(distance, k);
// #endif

  // Start timer
  time(&start);
    ParallelBitonicSort(data, label, distance, n);
//     BitonicSort(data, label, distance, n); // serial

#if DEBUG
  std::cout << "\ndata after sorting using parallel bitonic sort:\n";
  DisplayArray(data, k);
  DisplayArray(label, k);
  std::cout << "\ndistance after sorting using parallel bitonic sort:\n";
  DisplayArray(distance, k);
    
  int mode_value = mode_fn(label, k);
    
  std::cout << "Query value: " << query << ", Classification result: " << mode_value << std::endl;
#endif
    
  // Verify bitonic sort algorithms
  bool pass = true;
  for (int i = 0; i < size - 1; i++) {
    // Validate the sequence order is increasing
      if ((distance[i] > distance[i+1])) {
          pass = false;
      break;
    }
  }
  time(&end);
  std::cout << "Main | start: " << start << ", end: " << end << ", duration: " << end - start << " seconds " << std::endl;

  // Clean memory.
    free(data);
    free(label);
    free(distance);

  if (!pass) {
    std::cout << "\nFailed!\n";
    return -2;
  }

  std::cout << "\nSuccess!\n";
  return 0;
}