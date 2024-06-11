

from Meg import Meg
from Meg import Mip


if __name__ == "__main__":
    
    # window = Meg().run()

    eq = Mip.Line.parse_equation("( (-2 + y) / 3 )**(1/2)")
    # print(res)

    for i in range(10):
        res = eq.result(i)
        print(res)


    
    eq = Mip.Line.parse_equation("y**2 + 2 * y + 1")
    # print(res)

    for i in range(10):
        res = eq.result(i)
        print(res)


