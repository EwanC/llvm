// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: *

// Test updating a graph in a threaded situation

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

  auto dataA2 = dataA;
  auto dataB2 = dataB;
  auto dataC2 = dataC;

  {
    ext::oneapi::experimental::command_graph graphA{testQueue.get_context(),
                                                    testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    graphA.begin_recording(testQueue);

    // Record commands to graph
    run_kernels(testQueue, size, bufferA, bufferB, bufferC);

    graphA.end_recording();

    auto graphExec = graphA.finalize();

    ext::oneapi::experimental::command_graph graphB{testQueue.get_context(),
                                                    testQueue.get_device()};

    buffer<T> bufferA2{dataA2.data(), range<1>{dataA2.size()}};
    buffer<T> bufferB2{dataB2.data(), range<1>{dataB2.size()}};
    buffer<T> bufferC2{dataC2.data(), range<1>{dataC2.size()}};

    graphB.begin_recording(testQueue);

    // Record commands to graph
    run_kernels(testQueue, size, bufferA2, bufferB2, bufferC2);

    graphB.end_recording();

    auto updateGraph = [&]() { graphExec.update(graphB); };

    std::vector<std::thread> threads;
    threads.reserve(iterations);

    for (unsigned i = 0; i < iterations; ++i) {
      threads.emplace_back(updateGraph);
    }

    for (unsigned i = 0; i < iterations; ++i) {
      threads[i].join();
    }
  }

  return 0;
}