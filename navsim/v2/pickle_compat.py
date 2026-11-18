"""Read trusted local v2 caches without binding their classes to the v1 runtime.

This is namespace compatibility, not a security sandbox. Never load untrusted pickle.
"""

import io
import pickle


class V2Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("navsim.") and not module.startswith("navsim.v2."):
            module = "navsim.v2." + module[len("navsim.") :]
        return super().find_class(module, name)


def load(handle):
    return V2Unpickler(handle).load()


def loads(data):
    return load(io.BytesIO(data))
