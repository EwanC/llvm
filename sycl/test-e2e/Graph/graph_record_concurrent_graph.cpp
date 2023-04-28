// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** Tests attempting to record to a command_graph when it is already being
 * recorded to by another queue.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  bool success = false;

  ext::oneapi::experimental::command_graph graph{testQueue.get_context(),
                                                 testQueue.get_device()};
  graph.begin_recording(testQueue);

  queue testQueue2;
  try {
    graph.begin_recording(testQueue2);
  } catch (sycl::exception &e) {
    auto stdErrc = e.code().value();
    if (stdErrc == static_cast<int>(errc::invalid)) {
      success = true;
    }
  }

  graph.end_recording();
  assert(success);
  return 0;
}
