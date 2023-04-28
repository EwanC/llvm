// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test recording commands to a queue in a threaded situation. We don't
// submit the graph to verify the results as ordering of graph nodes isn't
// defined.

#include "graph_common.hpp"

using namespace sycl;

#include <thread>

int main() {
  queue testQueue;

  using T = int;

  const unsigned iterations = std::thread::hardware_concurrency();
  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  {
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        graph{testQueue.get_context(), testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    graph.begin_recording(testQueue);
    auto recordGraph = [&]() {
      // Record commands to graph
      run_kernels(testQueue, size, bufferA, bufferB, bufferC);
    };
    graph.end_recording();

    std::vector<std::thread> threads;
    threads.reserve(iterations);
    for (unsigned i = 0; i < iterations; ++i) {
      threads.emplace_back(recordGraph);
    }

    for (unsigned i = 0; i < iterations; ++i) {
      threads[i].join();
    }
  }

  return 0;
}