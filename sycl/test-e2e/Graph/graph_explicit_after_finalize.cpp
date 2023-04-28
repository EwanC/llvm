// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** This test creates a graph, finalizes it, then to adds a new nodes
 *  with the explicit API to the graph before finalizing and executing the
 *  second graph.
 */

#include "graph_common.hpp"

using namespace sycl;

class vector_plus_equals;
class write_to_output;

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size), dataOut(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);
  std::iota(dataOut.begin(), dataOut.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceC(dataC);
  std::vector<T> referenceOut(dataOut);
  for (unsigned n = 0; n < iterations * 2; n++) {
    for (size_t i = 0; i < size; i++) {
      referenceC[i] += (dataA[i] + dataB[i]);
      if (n >= iterations)
        referenceOut[i] += referenceC[i] + 1;
    }
  }

  {
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        graph{testQueue.get_context(), testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};
    buffer<T> bufferOut{dataOut.data(), range<1>{dataOut.size()}};

    // Vector add to some buffer
    graph.add([&](handler &cgh) {
      auto ptrA = bufferA.get_access<access::mode::read>(cgh);
      auto ptrB = bufferB.get_access<access::mode::read>(cgh);
      auto ptrC = bufferC.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<vector_plus_equals>(
          range<1>(size), [=](item<1> id) { ptrC[id] += ptrA[id] + ptrB[id]; });
    });

    auto graphExec = graph.finalize();

    // Read and modify previous output and write to output buffer
    graph.add([&](handler &cgh) {
      auto ptrC = bufferC.get_access<access::mode::read>(cgh);
      auto ptrOut = bufferOut.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<write_to_output>(
          range<1>(size), [=](item<1> id) { ptrOut[id] += ptrC[id] + 1; });
    });

    // Finalize a graph with the additional kernel for writing out to
    auto graphExecAdditional = graph.finalize();

    // Execute several iterations of the graph
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    }
    // Execute the extended graph.
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit(
          [&](handler &cgh) { cgh.ext_oneapi_graph(graphExecAdditional); });
    }
    // Perform a wait on all graph submissions.
    testQueue.wait_and_throw();
  }

  assert(referenceC == dataC);
  assert(referenceOut == dataOut);

  return 0;
}
