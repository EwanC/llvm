// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test each thread adding of nodes to same graph

#include "graph_common.hpp"

#include <thread>

using namespace sycl;

int main() {
  queue testQueue;

  using T = int;

  const size_t size = 1024;
  const unsigned iterations = std::thread::hardware_concurrency();
  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  {
    ext::oneapi::experimental::command_graph graph{testQueue.get_context(),
                                                   testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    auto AddNodesToGraph = [&]() {
      // Add commands to graph
      run_kernels(graph, size, bufferA, bufferB, bufferC);
    };

    std::vector<std::thread> threads;
    threads.reserve(iterations);
    for (unsigned i = 0; i < iterations; ++i) {
      threads.emplace_back(AddNodesToGraph);
    }

    for (unsigned i = 0; i < iterations; ++i) {
      threads[i].join();
    }
  }

  return 0;
}
