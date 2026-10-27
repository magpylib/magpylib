"""Field computations in chunks of kernel rows, run on threads: shared by
getB, getH, getJ, getM and the functional interface."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from magpylib._src.defaults.defaults_classes import default_settings

# Kernel rows per chunk: one row is one source, path step and observer, times
# the segments or faces the source is computed from. Every input is tiled to
# the rows before the kernel runs, and kernels hold several temporaries per
# row, so memory grows with this number; above it, a computation runs in chunks.
CHUNK_ROWS = 50_000


def run_chunks(tasks, rows):
    """Run every ``(compute, chunk)`` task of a computation of ``rows`` kernel
    rows: on ``defaults.threads`` threads above CHUNK_ROWS, else on this
    thread. NumPy releases the GIL inside most kernels, so the threads run in
    parallel. The chunks do not depend on the thread count, so neither do the
    results."""
    threads = min(default_settings.threads, len(tasks)) if rows > CHUNK_ROWS else 1
    if threads > 1:
        # a new thread starts from NumPy's default error settings
        errstate, errcall = np.geterr(), np.geterrcall()

        def run(task):
            compute, chunk = task
            with np.errstate(call=errcall, **errstate):
                compute(chunk)

        try:
            with ThreadPoolExecutor(threads) as pool:
                futures = [pool.submit(run, task) for task in tasks]
        except RuntimeError:  # no thread starts, e.g. at interpreter shutdown
            pass
        else:
            for future in futures:
                future.result()  # re-raises a chunk's error
            return
    for compute, chunk in tasks:
        compute(chunk)
