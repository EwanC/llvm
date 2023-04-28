// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** This test creates a graph, finalizes it, then submits that as a subgraph of
 * another graph and executes that second graph.
 */

#include "graph_common.hpp"

using namespace sycl;

class sub_vec_add_kernel;
class sub_subtract_kernel;
class mod_input_kernel;
class copy_out_kernel;

int main() {
  queue testQueue;

  using T = int;

  // Values used to modify data inside kernels.
  const int mod_value = 7;
  std::vector<T> dataA(size), dataB(size), dataC(size), dataOut(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);
  std::iota(dataOut.begin(), dataOut.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceA(dataA);
  std::vector<T> referenceB(dataB);
  std::vector<T> referenceC(dataC);
  std::vector<T> referenceOut(dataOut);
  for (unsigned n = 0; n < iterations; n++) {
    for (size_t i = 0; i < size; i++) {
      referenceA[i] += mod_value;
      referenceB[i] += mod_value;
      referenceC[i] = (referenceA[i] + referenceB[i]);
      referenceC[i] -= mod_value;
      referenceOut[i] = referenceC[i] + mod_value;
    }
  }

  {
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        subGraph{testQueue.get_context(), testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};
    buffer<T> bufferOut{dataOut.data(), range<1>{dataOut.size()}};

    // Record some operations to a graph which will later be submitted as part
    // of another graph.
    subGraph.begin_recording(testQueue);

    // Vector add two values
    testQueue.submit([&](handler &cgh) {
      auto ptrA = bufferA.get_access<access::mode::read>(cgh);
      auto ptrB = bufferB.get_access<access::mode::read>(cgh);
      auto ptrC = bufferC.get_access<access::mode::write>(cgh);
      cgh.parallel_for<sub_vec_add_kernel>(
          range<1>(size), [=](item<1> id) { ptrC[id] = ptrA[id] + ptrB[id]; });
    });

    // Modify the output value with some other value
    testQueue.submit([&](handler &cgh) {
      auto ptrC = bufferC.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sub_subtract_kernel>(
          range<1>(size), [=](item<1> id) { ptrC[id] -= mod_value; });
    });

    subGraph.end_recording();

    auto subGraphExec = subGraph.finalize();

    ext::oneapi::experimental::command_graph mainGraph{testQueue.get_context(),
                                                       testQueue.get_device()};

    mainGraph.begin_recording(testQueue);

    // Modify the input values.
    testQueue.submit([&](handler &cgh) {
      auto ptrA = bufferA.get_access<access::mode::read_write>(cgh);
      auto ptrB = bufferB.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<mod_input_kernel>(range<1>(size), [=](item<1> id) {
        ptrA[id] += mod_value;
        ptrB[id] += mod_value;
      });
    });

    testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(subGraphExec); });

    // Copy to another output buffer.
    testQueue.submit([&](handler &cgh) {
      auto ptrC = bufferC.get_access<access::mode::read>(cgh);
      auto ptrOut = bufferOut.get_access<access::mode::write>(cgh);
      cgh.parallel_for<copy_out_kernel>(range<1>(size), [=](item<1> id) {
        ptrOut[id] = ptrC[id] + mod_value;
      });
    });

    mainGraph.end_recording();

    // Finalize a graph with the additional kernel for writing out to
    auto mainGraphExec = mainGraph.finalize();

    // Execute several iterations of the graph
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit(
          [&](handler &cgh) { cgh.ext_oneapi_graph(mainGraphExec); });
    }
    // Perform a wait on all graph submissions.
    testQueue.wait_and_throw();
  }

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);
  assert(referenceOut == dataOut);
}
