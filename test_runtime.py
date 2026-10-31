"""Check the local runtime without filling in any exercise solutions."""

import unittest
from contextlib import redirect_stdout
from io import StringIO

from lib import Coord, CudaProblem
import numba
import numpy as np


class RuntimeTests(unittest.TestCase):
    def test_cpu_kernel_with_shared_memory(self):
        def kernel(cuda):
            def call(out, values, size):
                shared = cuda.shared.array(4, numba.float32)
                local = cuda.threadIdx.x
                index = cuda.blockIdx.x * cuda.blockDim.x + local
                shared[local] = values[index] if index < size else 0
                cuda.syncthreads()
                if index < size:
                    out[index] = shared[3 - local]

            return call

        self.assertTrue(numba.config.ENABLE_CUDASIM)
        values = np.arange(7, dtype=np.float32)
        problem = CudaProblem(
            "Runtime check", kernel, [values], np.zeros(7), [7],
            blockspergrid=Coord(2, 1), threadsperblock=Coord(4, 1),
            spec=lambda a: np.array([3, 2, 1, 0, 0, 6, 5]),
        )
        np.testing.assert_array_equal(problem.run_cuda(), problem.spec(values))
        with redirect_stdout(StringIO()) as output:
            problem.check()
        self.assertIn("Passed Tests!", output.getvalue())

    def test_notebook_diagram(self):
        def kernel(cuda):
            def call(out, values):
                out[cuda.threadIdx.x] = values[cuda.threadIdx.x]

            return call

        problem = CudaProblem(
            "Diagram check", kernel, [np.arange(2)], np.zeros(2),
            threadsperblock=Coord(2, 1),
        )
        with redirect_stdout(StringIO()):
            svg = problem.show()._repr_svg_()
        self.assertIn("<svg", svg)
        self.assertIn("Diagram check", svg)


if __name__ == "__main__":
    unittest.main()
