import typing as ty
from pydantic.generics import GenericModel


T = ty.TypeVar("T")
CT: ty.TypeAlias = tuple[bool, T, str]

class CT2(ty.Generic[T], tuple[bool, T, str]):
    pass

class MyClass(ty.Generic[T]):
    internal1: tuple[bool, T, str]
    internal2: CT2[T]
    internal3: CT2[float]


class DerivedClass(MyClass[float]):
    pass

print(ty.get_type_hints(MyClass))
print(ty.get_type_hints(DerivedClass))

x = DerivedClass()
x.internal2