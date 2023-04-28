// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** Tests recording and submission of a graph containing usm memcpy commands.
 */

#include "graph_common.hpp"

using namespace sycl;

class kernel_mod_a;
class kernel_mod_b;

int main() {
  queue testQueue;

  using T = int;

  const T modValue = 7;
  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  for (size_t i = 0; i < iterations; i++) {
    for (size_t j = 0; j < size; j++) {
      referenceA[j] = referenceB[j];
      referenceA[j] += modValue;
      referenceB[j] = referenceA[j];
      referenceB[j] += modValue;
      referenceC[j] = referenceB[j];
    }
  }

  ext::oneapi::experimental::command_graph<
      ext::oneapi::experimental::graph_state::modifiable>
      graph{testQueue.get_context(), testQueue.get_device()};

  {
    auto ptrA = malloc_device<T>(dataA.size(), testQueue);
    testQueue.memcpy(ptrA, dataA.data(), dataA.size() * sizeof(T)).wait();
    auto ptrB = malloc_device<T>(dataB.size(), testQueue);
    testQueue.memcpy(ptrB, dataB.data(), dataB.size() * sizeof(T)).wait();
    auto ptrC = malloc_device<T>(dataC.size(), testQueue);
    testQueue.memcpy(ptrC, dataC.data(), dataC.size() * sizeof(T)).wait();

    graph.begin_recording(testQueue);

    // Record commands to graph
    // memcpy from B to A
    auto eventA = testQueue.copy(ptrB, ptrA, size);
    // Read & write A
    auto eventB = testQueue.submit([&](handler &cgh) {
      cgh.depends_on(eventA);
      cgh.parallel_for<kernel_mod_a>(range<1>(size), [=](item<1> id) {
        auto linID = id.get_linear_id();
        ptrA[linID] += modValue;
      });
    });

    // memcpy from A to B
    auto eventC = testQueue.copy(ptrA, ptrB, size, eventB);

    // Read and write B
    auto eventD = testQueue.submit([&](handler &cgh) {
      cgh.depends_on(eventC);
      cgh.parallel_for<kernel_mod_b>(range<1>(size), [=](item<1> id) {
        auto linID = id.get_linear_id();
        ptrB[linID] += modValue;
      });
    });

    // memcpy from B to C
    testQueue.copy(ptrB, ptrC, size, eventD);

    graph.end_recording();
    auto graphExec = graph.finalize();

    // Execute graph over n iterations
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    }
    // Perform a wait on all graph submissions.
    testQueue.wait();

    testQueue.memcpy(dataA.data(), ptrA, dataA.size() * sizeof(T)).wait();
    testQueue.memcpy(dataB.data(), ptrB, dataB.size() * sizeof(T)).wait();
    testQueue.memcpy(dataC.data(), ptrC, dataC.size() * sizeof(T)).wait();

    free(ptrA, testQueue.get_context());
    free(ptrB, testQueue.get_context());
    free(ptrC, testQueue.get_context());
  }

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  return 0;
}
