// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: *

/** Tests whole graph update by creating a double buffering scenario, where a
 * single graph is repeatedly executed then updated to swap between two sets of
 * buffers.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size);
  std::vector<T> dataA2(size), dataB2(size), dataC2(size);
  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  std::iota(dataA2.begin(), dataA2.end(), 3);
  std::iota(dataB2.begin(), dataB2.end(), 13);
  std::iota(dataC2.begin(), dataC2.end(), 1333);

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  std::vector<T> referenceA2(dataA2), referenceB2(dataB2), referenceC2(dataC2);
  // Calculate reference data
  calculate_reference_data(iterations, size, referenceA, referenceB,
                           referenceC);
  calculate_reference_data(iterations, size, referenceA2, referenceB2,
                           referenceC2);

  {
    ext::oneapi::experimental::command_graph graph{testQueue.get_context(),
                                                   testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    buffer<T> bufferA2{dataA2.data(), range<1>{dataA2.size()}};
    buffer<T> bufferB2{dataB2.data(), range<1>{dataB2.size()}};
    buffer<T> bufferC2{dataC2.data(), range<1>{dataC2.size()}};

    add_kernels(graph, size, bufferA, bufferB, bufferC);

    auto execGraph = graph.finalize();

    // Create second graph using other buffer set
    ext::oneapi::experimental::command_graph graphUpdate{
        testQueue.get_context(), testQueue.get_device()};
    run_kernels(graphUpdate, size, bufferA2, bufferB2, bufferC2);

    for (size_t i = 0; i < iterations; i++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
      // Update to second set of buffers
      execGraph.update(graphUpdate);
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
      // Reset back to original buffers
      execGraph.update(graph);
    }

    testQueue.wait_and_throw();
  }

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  assert(referenceA2 == dataA2);
  assert(referenceB2 == dataB2);
  assert(referenceC2 == dataC2);

  return 0;
}
