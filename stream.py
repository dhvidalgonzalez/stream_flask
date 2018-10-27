import numpy as np
from timeit import default_timer as timer


def checktick():
    M = 20
    timesfound = np.empty((M,))

    for i in range(M):

        t1 = timer()
        t2 = timer()
        while (t2 - t1) < 1e-6:
            t2 = timer()
        t1 = t2
        timesfound[i] = t1

    minDelta = 1000000
    Delta = (1e6 * np.diff(timesfound)).astype(np.int)
    minDelta = Delta.min()
    return minDelta


def main(STREAM_ARRAY_SIZE, NTIMES, OFFSET, STREAM_TYPE, tests, desc):

    """STREAM_ARRAY_SIZE = args.STREAM_ARRAY_SIZE
    NTIMES = args.NTIMES
    OFFSET = args.OFFSET
    STREAM_TYPE = args.STREAM_TYPE"""

    HLINE = "-------------------------------------------------------------"

    a = np.empty((STREAM_ARRAY_SIZE+OFFSET,), dtype=STREAM_TYPE)
    b = np.empty((STREAM_ARRAY_SIZE+OFFSET,), dtype=STREAM_TYPE)
    c = np.empty((STREAM_ARRAY_SIZE+OFFSET,), dtype=STREAM_TYPE)

    avgtime = np.zeros((4,), dtype='double')
    maxtime = np.zeros((4,), dtype='double')
    FLT_MAX = np.finfo('single').max
    mintime = np.array([FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX])

    label = ["Copy:      ", "Scale:     ", "Add:       ", "Triad:     "]

    tbytes = np.array([2 * np.nbytes[STREAM_TYPE] * STREAM_ARRAY_SIZE,
                       2 * np.nbytes[STREAM_TYPE] * STREAM_ARRAY_SIZE,
                       3 * np.nbytes[STREAM_TYPE] * STREAM_ARRAY_SIZE,
                       3 * np.nbytes[STREAM_TYPE] * STREAM_ARRAY_SIZE],
                      dtype='double')

    quantum = 0
    BytesPerWord = 0
    times = np.zeros((4, NTIMES))

    # --- SETUP --- determine precision and check timing ---
    str_output = ''

    str_output += HLINE + '<br>'
    str_output += "pySTREAM version 0.2 <br>"
    str_output += HLINE + '<br>'
    BytesPerWord = np.nbytes[STREAM_TYPE]
    str_output += "Bytes per array element: %d <br>" % BytesPerWord
    str_output += "             Array size: %d (elements) <br>" % STREAM_ARRAY_SIZE
    str_output += "                 Offset: %d (elements) <br>" % OFFSET
    str_output += "       Memory per array: %.2f MiB (= %.2f GiB). <br>" %\
          (BytesPerWord * (STREAM_ARRAY_SIZE / 1024.0 / 1024.0),\
           BytesPerWord * (STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0))
    str_output += "  Total memory required: %.2f MiB (= %.2f GiB). <br>" %\
          (3.0 * BytesPerWord * (STREAM_ARRAY_SIZE / 1024.0 / 1024.0),\
           3.0 * BytesPerWord * (STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.))
    str_output += "        Number of tests: %d <br>" % NTIMES

    # Get initial value for system clock.
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = 1.0
        b[j] = 2.0
        c[j] = 0.0

    quantum = checktick()
    if quantum >= 1:
        str_output += "      Clock granularity: ~%d us <br>" % quantum
    else:
        str_output += "      Clock granularity: <1 us <br>" % quantum
        quantum = 1

    t = timer()
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = 2.0 * a[j]
    t = 1.0e6 * (timer() - t)

    str_output += "              Test time: ~%d us <br>" % int(t)
    str_output += "                       :  %d clock ticks) <br>" % int(t/quantum)
    if int(t/quantum) <= 20:
        str_output += "Note -- this should be > 20 clock ticks <br>"

    str_output += HLINE + '<br>'
    str_output += "Note -- Bandwidth is calculated using the *minimum* time <br>"
    str_output += "        (after the first iteration) <br>"
    str_output += "Note -- This is only a guideline. <br>"
    str_output += HLINE + '<br>'

    # --- MAIN LOOP --- repeat test cases NTIMES times ---

    scalar = 3.0
    for test in tests:

        if test in desc:
            str_output += '## %s <br>' % desc[test]

        times[:] = 0.0
        for k in range(NTIMES):

            if test == 'reference':
                times[0][k] = timer()
                for j in range(STREAM_ARRAY_SIZE):
                    c[j] = a[j]
                times[0][k] = timer() - times[0][k]

                times[1][k] = timer()
                for j in range(STREAM_ARRAY_SIZE):
                    b[j] = scalar*c[j]
                times[1][k] = timer() - times[1][k]

                times[2][k] = timer()
                for j in range(STREAM_ARRAY_SIZE):
                    c[j] = a[j]+b[j]
                times[2][k] = timer() - times[2][k]

                times[3][k] = timer()
                for j in range(STREAM_ARRAY_SIZE):
                    a[j] = b[j]+scalar*c[j]
                times[3][k] = timer() - times[3][k]

            elif test == 'vector':
                times[0][k] = timer()
                c[:] = a[:]
                times[0][k] = timer() - times[0][k]

                times[1][k] = timer()
                b[:] = scalar * c[:]
                times[1][k] = timer() - times[1][k]

                times[2][k] = timer()
                c[:] = a[:] + b[:]
                times[2][k] = timer() - times[2][k]

                times[3][k] = timer()
                a[:] = b[:] + scalar * c[:]
                times[3][k] = timer() - times[3][k]

            elif test == 'numpyops':
                times[0][k] = timer()
                c = a.copy()
                times[0][k] = timer() - times[0][k]

                times[1][k] = timer()
                c *= scalar
                b = c.copy()
                times[1][k] = timer() - times[1][k]

                times[2][k] = timer()
                c = a + b
                times[2][k] = timer() - times[2][k]

                times[3][k] = timer()
                c *= scalar
                a = b + c
                times[3][k] = timer() - times[3][k]

            elif test == 'cython_ref' or test == 'cython_omp':

                if test == 'cython_ref':
                    from cython_ref import xcopy, xscale, xadd, xtriad
                if test == 'cython_omp':
                    from cython_omp import xcopy, xscale, xadd, xtriad

                times[0][k] = timer()
                xcopy(a, c)
                times[0][k] = timer() - times[0][k]

                times[1][k] = timer()
                xscale(b, c, scalar)
                times[1][k] = timer() - times[1][k]

                times[2][k] = timer()
                xadd(a, b, c)
                times[2][k] = timer() - times[2][k]

                times[3][k] = timer()
                xtriad(a, b, c, scalar)
                times[3][k] = timer() - times[3][k]

            elif test == 'pybind11_ref':

                if test == 'pybind11_ref':
                    import pybind11_ref

                times[0][k] = timer()
                pybind11_ref.copy(a, c)
                times[0][k] = timer() - times[0][k]

                times[1][k] = timer()
                pybind11_ref.scale(b, c, scalar)
                times[1][k] = timer() - times[1][k]

                times[2][k] = timer()
                pybind11_ref.add(a, b, c)
                times[2][k] = timer() - times[2][k]

                times[3][k] = timer()
                pybind11_ref.triad(a, b, c, scalar)
                times[3][k] = timer() - times[3][k]

            else:
                str_output += '...test not implemented <br>'

        # --- SUMMARY ---

        avgtime = times[:, 1:].mean(axis=1)  # note -- skip first iteration
        mintime = times[:, 1:].min(axis=1)
        maxtime = times[:, 1:].max(axis=1)

        str_output += "``` <br>"
        str_output += "Function    Best Rate GB/s  Avg time     Min time     Max time <br>"
        for j in range(4):
            str_output += "%s%12.1f  %11.6f  %11.6f  %11.6f <br>" %\
                  (label[j],\
                   1.0e-09 * tbytes[j]/mintime[j],\
                   avgtime[j],\
                   mintime[j],\
                   maxtime[j])
        str_output += "```<br>"

        return str_output
