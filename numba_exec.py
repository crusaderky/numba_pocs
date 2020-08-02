import hashlib
import pickle
import numba.core.caching


def exec_with_numba_cache(source: str) -> dict:
    source_stamp = hashlib.sha1(source.encode("utf-8")).hexdigest()
    globals_ = {"_exec_source_stamp": source_stamp}
    fname = f"<{source_stamp}>"
    code = compile(source, filename=fname, mode="exec", dont_inherit=True)
    exec(code, globals_)
    del globals_["_exec_source_stamp"]
    return globals_


class _ExecCacheLocator(numba.core.caching._CacheLocator):
    def __init__(self, source_stamp: str, disambiguator: str):
        self.source_stamp = source_stamp
        self.disambiguator = disambiguator

    def get_cache_path(self):
        return "mycache"

    def get_source_stamp(self):
        return self.source_stamp

    def get_disambiguator(self):
        return self.disambiguator

    @classmethod
    def from_function(cls, py_func, py_file):
        try:
            source_stamp = py_func.__globals__["_exec_source_stamp"]
        except KeyError:
            return None

        # Avoid this code path:
        # https://github.com/numba/numba/blob/ccdf61381cc543afda76a80ef4c51e613472e1f7/numba/core/funcdesc.py#L152-L157  # noqa: E501
        py_func.__module__ = "__main__"

        # Create a unique hash of the function, in case of multiple functions with the
        # same name are in the same source code. It will be appended to the file names.
        co = {
            attr: getattr(py_func.__code__, attr)
            for attr in dir(py_func.__code__)
            if attr.startswith("co_") and attr != "co_filename"
        }
        disambiguator = hashlib.sha1(pickle.dumps(co)).hexdigest()[:10]
        return cls(source_stamp, disambiguator)


numba.core.caching._CacheImpl._locator_classes.append(_ExecCacheLocator)
