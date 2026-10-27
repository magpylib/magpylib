"""Field computations in chunks of kernel rows, run on threads: shared by
getB, getH, getJ, getM and the functional interface."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from magpylib._src.defaults.defaults_classes import default_settings

# Kernel rows per chunk: one row is one source, path step and observer, times
# the segments or faces the source is computed from. Every input is tiled to
# the rows before the kernel runs, and kernels hold several temporaries per
# row, so memory grows with this number; above it, a computation runs in chunks.
CHUNK_ROWS = 50_000


def run_chunks(tasks, rows, largest_chunk=0):
    """Run every ``(compute, chunk)`` task of a computation of ``rows`` kernel
    rows: on ``defaults.threads`` threads above CHUNK_ROWS, else on this
    thread. A largest chunk above CHUNK_ROWS (one observer of a source with
    more faces) runs on fewer threads, so that memory stays near threads
    times CHUNK_ROWS rows. NumPy releases the GIL inside most kernels, so the
    threads run in parallel. The chunks do not depend on the thread count, so
    neither do the results."""
    threads = min(default_settings.threads, len(tasks)) if rows > CHUNK_ROWS else 1
    threads = max(1, threads * CHUNK_ROWS // max(largest_chunk, CHUNK_ROWS))
    rest = tasks
    if threads > 1:
        # a new thread starts from NumPy's default error settings
        errstate, errcall = np.geterr(), np.geterrcall()

        def run(task):
            compute, chunk = task
            with np.errstate(call=errcall, **errstate):
                compute(chunk)

        pool = ThreadPoolExecutor(threads)
        futures = []
        try:
            try:
                for task in tasks:  # one by one: a failed submit leaves the rest
                    futures.append(pool.submit(run, task))  # noqa: PERF401
                rest = []
            except RuntimeError:  # no more threads start, e.g. at interpreter shutdown
                rest = tasks[len(futures) :]  # computed on this thread below
            for future in as_completed(futures):
                future.result()  # re-raises a chunk's error
        finally:  # after an error or Ctrl+C, the queued chunks are dropped
            pool.shutdown(cancel_futures=True)
    for compute, chunk in rest:
        compute(chunk)
