import inspect
from types import FunctionType


class BadSignatureException(Exception):
    pass


class SignatureCheckerMeta(type):
    def __new__(cls, name, baseClasses, d):
        #For each method in d, check to see if any base class already
        #defined a method with that name. If so, make sure the
        #signatures are the same.
        for methodName in d:
            func = d[methodName]

            if not isinstance(func, FunctionType):
                continue
            for baseClass in baseClasses:
                try:
                    fBase = getattr(baseClass, methodName)
                    if not inspect.signature(func) == inspect.signature(fBase):
                        raise BadSignatureException(f"{methodName}")
                except AttributeError:
                    #This method was not defined in this base class,
                    #So just go to the next base class.
                    continue

        return super().__new__(cls, name, baseClasses, d)
