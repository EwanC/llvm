// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test finalizing and submitting a graph in a threaded situation

#include "graph_common.hpp"

#include <thread>

using namespace sycl;

int main() {
  queue testQueue;

  using T = int;

  const unsigned iterations = std::thread::hardware_concurrency();
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
    auto finalizeGraph = [&]() {
      auto graphExec = graph.finalize();
      testQueue.submit(
          [&](sycl::handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    };

    std::vector<std::thread> threads;
    threads.reserve(iterations);

    for (unsigned i = 0; i < iterations; ++i) {
      threads.emplace_back(finalizeGraph);
    }

    for (unsigned i = 0; i < iterations; ++i) {
      threads[i].join();
    }

    // Perform a wait on all graph submissions.
    testQueue.wait();
  }

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  return 0;
}
