// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test finalizing and submitting a graph in a threaded situation

#include "graph_common.hpp"

#include <thread>

using namespace sycl;

int main() {
  queue testQueue;

  const unsigned iterations = std::thread::hardware_concurrency();

  {
    auto recordGraph = [&]() {
      ext::oneapi::experimental::command_graph graph{testQueue.get_context(),
                                                     testQueue.get_device()};
      graph.begin_recording(testQueue);
      graph.end_recording();
    };

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