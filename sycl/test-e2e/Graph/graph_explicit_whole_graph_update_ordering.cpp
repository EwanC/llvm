// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: *

/** Tests whole graph update by introducing a delay in to the update
 * transactions dependencies to check correctness of behaviour.
 */

#include "graph_common.hpp"

using namespace sycl;

class host_task_test;

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size);
  std::vector<T> hostTaskOutput(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  auto dataA2 = dataA;
  auto dataB2 = dataB;
  auto dataC2 = dataC;

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  calculate_reference_data(iterations, size, referenceA, referenceB,
                           referenceC);

  {
    ext::oneapi::experimental::command_graph graphA{testQueue.get_context(),
                                                    testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    buffer<T> hostTaskOutputBuffer{hostTaskOutput};

    // Add commands to graph
    add_kernels(graphA, size, bufferA, bufferB, bufferC);

    // host task to induce a wait for dependencies
    graphA.add([&](handler &cgh) {
      // This should be access::target::host_task but it has not been
      // implemented yet.
      auto ptrIn = bufferC.get_access<access::mode::read_write,
                                      access::target::host_buffer>(cgh);
      auto ptrOut =
          hostTaskOutputBuffer.get_access<access::mode::read_write,
                                          access::target::host_buffer>(cgh);
      cgh.host_task([=]() {
        for (size_t i = 0; i < size; i++) {
          ptrOut[i] = ptrIn[i];
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      });
    });

    auto graphExec = graphA.finalize();

    ext::oneapi::experimental::command_graph graphB{testQueue.get_context(),
                                                    testQueue.get_device()};

    buffer<T> bufferA2{dataA2.data(), range<1>{dataA2.size()}};
    buffer<T> bufferB2{dataB2.data(), range<1>{dataB2.size()}};
    buffer<T> bufferC2{dataC2.data(), range<1>{dataC2.size()}};

    // Record commands to graph
    run_kernels(graphB, size, bufferA2, bufferB2, bufferC2);

    // host task to match the graph topology, but we don't need to sleep this
    // time because there is no following update.
    graphB.add([&](handler &cgh) {
      // This should be access::target::host_task but it has not been
      // implemented yet.
      auto ptrIn = bufferC2.get_access<access::mode::read_write,
                                       access::target::host_buffer>(cgh);
      auto ptrOut =
          hostTaskOutputBuffer.get_access<access::mode::read_write,
                                          access::target::host_buffer>(cgh);
      cgh.host_task([=]() {
        for (size_t i = 0; i < size; i++) {
          ptrOut[i] = ptrIn[i];
        }
      });
    });

    // Execute several iterations of the graph for 1st set of buffers
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    }

    graphExec.update(graphB);

    // Execute several iterations of the graph for 2nd set of buffers
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    }

    // Perform a wait on all graph submissions.
    testQueue.wait();
  }

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);
  assert(referenceC == hostTaskOutput);

  assert(referenceA == dataA2);
  assert(referenceB == dataB2);
  assert(referenceC == dataC2);

  return 0;
}
