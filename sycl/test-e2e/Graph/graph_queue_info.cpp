// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: *

/**  Tests the return values from queue graph functions which change the
 * internal queue state
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  ext::oneapi::experimental::queue_state state =
      testQueue.get_info<info::queue::state>();
  assert(state == ext::oneapi::experimental::queue_state::executing);

  ext::oneapi::experimental::command_graph<
      ext::oneapi::experimental::graph_state::modifiable>
      graph{testQueue.get_context(), testQueue.get_device()};
  graph.begin_recording(testQueue);
  state = testQueue.get_info<info::queue::state>();
  assert(state == ext::oneapi::experimental::queue_state::recording);

  graph.end_recording();
  state = testQueue.get_info<info::queue::state>();
  assert(state == ext::oneapi::experimental::queue_state::executing);

  return 0;
}
