# Dictionary
movie = {"name": "Lotr", "year": 2001}


# Typed Dictionary
from typing import TypedDict

class Movie(TypedDict):
    name: str
    year: int

movie = Movie(name="Lotr", year=2001)


# Union
from typing import Union
def square(x: Union[int, float]) -> float:
    return x * x

x=5
x=5.5
x="5" # This will raise a type error


# Optional
from typing import Optional

def greet(name: Optional[str]) -> str:
    if name is None:
        return "Hello, World!"
    else:
        return f"Hello, {name}!"
    

# Any
from typing import Any

def print_value(value: Any) -> None:
    print(value)


# Lambda Function
square = lambda x: x * x
print(square(5))

nums = [1, 2, 3, 4, 5]
squared_nums = list(map(lambda x: x * x, nums))