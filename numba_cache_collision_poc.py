import multiprocessing
import shutil
from textwrap import dedent
from numba_exec import exec_with_numba_cache


def run(*ii):
    print("New process")
    for i in ii:
        print(i)
        source = dedent(
            """
            import numba

            @numba.guvectorize(
                ["f8,f8[:]"], "()->()", cache=True, nopython=True
            )
            def f(_, out):
                out[0] = {}
            """.format(
                i
            )
        )

        globals_ = exec_with_numba_cache(source)
        f = globals_["f"]
        out = f(1337)
        assert out == i, f"{out} != {i}"


def main():
    try:
        shutil.rmtree("mycache")
    except FileNotFoundError:
        pass

    ctx = multiprocessing.get_context("spawn")

    # 1 - cache miss
    # 1 - cache hit, created by the same process
    # 2 - cache miss
    p = ctx.Process(target=run, args=(1, 1, 2))
    p.start()
    p.join()
    # 1 - cache hit, created by another process
    # 1 - cache hit, created by another process
    # 2 - cache hit, created by another process
    # 3 - cache miss
    p = ctx.Process(target=run, args=(1, 1, 2, 3))
    p.start()
    p.join()


if __name__ == "__main__":
    main()
