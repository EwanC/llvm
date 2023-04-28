// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL:*

/** This test uses a host_task when explicitly adding a command_graph node.
 */

#include "graph_common.hpp"

using namespace sycl;

class host_task_add;
class host_task_inc;

int main() {
  queue testQueue;

  using T = int;

  const T modValue = T{7};
  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceC(dataC);
  for (unsigned n = 0; n < iterations; n++) {
    for (size_t i = 0; i < size; i++) {
      referenceC[i] += (dataA[i] + dataB[i]) + modValue + 1;
    }
  }

  {
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        graph{testQueue.get_context(), testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    // Vector add to output
    graph.add([&](handler &cgh) {
      auto ptrA = bufferA.get_access<access::mode::read>(cgh);
      auto ptrB = bufferB.get_access<access::mode::read>(cgh);
      auto ptrOut = bufferC.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<host_task_add>(range<1>(size), [=](item<1> id) {
        ptrOut[id] += ptrA[id] + ptrB[id];
      });
    });

    // Modify the output values in a host_task
    graph.add([&](handler &cgh) {
      // This should be access::target::host_task but it has not been
      // implemented yet.
      auto hostC = bufferC.get_access<access::mode::read_write,
                                      access::target::host_buffer>(cgh);
      cgh.host_task([=]() {
        for (size_t i = 0; i < size; i++) {
          hostC[i] += modValue;
        }
      });
    });

    // Modify temp buffer and write to output buffer
    graph.add([&](handler &cgh) {
      auto ptrOut = bufferC.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<host_task_inc>(range<1>(size),
                                      [=](item<1> id) { ptrOut[id] += 1; });
    });

    auto graphExec = graph.finalize();

    // Execute several iterations of the graph
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    }
    // Perform a wait on all graph submissions.
    testQueue.wait();
  }

  assert(referenceC == dataC);

  return 0;
}
