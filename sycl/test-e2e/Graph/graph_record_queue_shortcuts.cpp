// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** Tests queue shortcuts for executing a graph */

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

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  calculate_reference_data(iterations, size, referenceA, referenceB,
                           referenceC);

  {
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        graph{testQueue.get_context(), testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    graph.begin_recording(testQueue);

    // Record commands to graph

    run_kernels(testQueue, size, bufferA, bufferB, bufferC);

    graph.end_recording();
    auto graphExec = graph.finalize();

    // Execute several iterations of the graph using the different shortcuts
    event e = testQueue.ext_oneapi_graph(graphExec);

    assert(iterations > 2);
    const unsigned loop_iterations = iterations - 2;
    std::vector<event> events(loop_iterations);
    for (unsigned n = 0; n < loop_iterations; n++) {
      events[n] = testQueue.ext_oneapi_graph(graphExec, e);
    }

    testQueue.ext_oneapi_graph(graphExec, events).wait();
  }

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  return 0;
}
