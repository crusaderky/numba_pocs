import gc
import linecache
from textwrap import dedent

from pympler import muppy
from numba_exec import exec_with_numba_cache
import numba.core.dispatcher


def get_objects():
    linecache.cache.clear()
    numba.core.dispatcher.Dispatcher._memo.clear()
    numba.core.dispatcher.Dispatcher._recent.clear()
    gc.collect()
    return muppy.get_objects(include_frames=True)


def main():
    # Warm LLVM up
    source = dedent(
        """
        import numba
    
        @numba.guvectorize(
            ["f8,f8[:]"], "()->()", cache=True, nopython=True
        )
        def warmup(x, out):
            out[0] = x * 2
        """
    )
    exec_with_numba_cache(source)

    source = dedent(
        """
        import numba
    
        @numba.guvectorize(
            ["f8,f8[:]"], "()->()", cache=True, nopython=True
        )
        def f(x, out):
            out[0] = x
        """
    )

    old_objects = get_objects()
    exec_with_numba_cache(source)
    new_objects = get_objects()
    ignore_ids = {id(obj) for obj in old_objects} | {id(old_objects)}
    leaked = [obj for obj in new_objects if id(obj) not in ignore_ids]
    for obj in leaked:
        print(id(obj), type(obj))
    assert not leaked


if __name__ == "__main__":
    main()
