// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: *

/** Tests whole graph update by creating two graphs with different buffers and
 * attempting to update one from the other.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size);

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

    // Add commands to graph
    add(graphA, size, bufferA, bufferB, bufferC);
    auto graphExec = graphA.finalize();

    ext::oneapi::experimental::command_graph graphB{testQueue.get_context(),
                                                    testQueue.get_device()};

    buffer<T> bufferA2{dataA2.data(), range<1>{dataA2.size()}};
    buffer<T> bufferB2{dataB2.data(), range<1>{dataB2.size()}};
    buffer<T> bufferC2{dataC2.data(), range<1>{dataC2.size()}};

    // Record commands to graph
    add_kernels(graphB, size, bufferA2, bufferB2, bufferC2);

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

  assert(referenceA == dataA2);
  assert(referenceB == dataB2);
  assert(referenceC == dataC2);

  return 0;
}
