from abc import ABC, abstractmethod
import bisect
from cmath import cos, sin
import cmath
from math import floor
from random import randint
import re
from typing import Union
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import Qt, QRectF
import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtWidgets
import sys

class Meg():
    def run(self):
        self.app = QApplication(sys.argv)
        self.window = MegWindow()
        sys.exit(self.app.exec_())


DEFAULT_COLOR = "#1a1919"

class MegWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()

        
        # set needed variables
        self.width = screen_geometry.width() // 2
        self.height = screen_geometry.height() // 2


        # remove os frame and make the window transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Set the window size
        print(screen_geometry.width(), screen_geometry.height())
        self.setGeometry(0, 0, screen_geometry.width(), screen_geometry.height())

        # Show the window
        self.show()


    # mouse events
    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        print(f"Mouse pressed at ({x}, {y})")


    # method called by QT to paint the window
    def paintEvent(self, event):
        # make the window background with rounded corners
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtGui.QColor(DEFAULT_COLOR))  # RGBA color
        painter.setPen(QtGui.QPen(QtGui.QColor(DEFAULT_COLOR), 0))
        path = QtGui.QPainterPath()
        radius = 10
        path.addRoundedRect(QRectF(0, 38, self.width, self.height-38), radius, radius)
        painter.drawPath(path)




class Mip(ABC):
    """
    Mip is a image format inspired by SVG. This class parses the Mip format and stores the attributes
    internally. These attributes can then be converted to an image with the draw method which produces
    an array of pixel values. 

    Attributes:
        width (int): The width of the Mip.
        height (int): The height of the Mip.
        color (str): The color of the Mip.
    """

    DEFAULT_BACKGROUND = "rgba(0, 0, 0, 0)"
    DEFAULT_COLOR = "#1a1919"


    class Operator():
        """
        An operator in an algebraic equation or conditional expression. This can be +, -, *, /, **; for
        algebra, and ==, !=, <, >, and, or; for conditional expressions.
        """
        
        OPERATORS = {'+', '-', '*', '/', '**', '==,' '!=', '<', '>', 'and', 'or'}
        OPERATOR_PRECEDENCE = {
            '==': 1,
            '!=': 1,
            '<': 1,
            '>': 1,
            'and': 1,
            'or': 1,
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '**': 3
        }

        def __init__(
            self,
            operator: str
        ):
            self.operator = operator
        
        def apply(self, a: float, b: float):
            if self.operator == "+":
                return a + b
            elif self.operator == "-":
                return a - b
            elif self.operator == "*":
                return a * b
            elif self.operator == "/":
                return a / b
            elif self.operator == "**":
                return a ** b
            elif self.operator == "==":
                return a == b
            elif self.operator == "!=":
                return a != b
            elif self.operator == "<":
                return a < b
            elif self.operator == ">":
                return a > b
            elif self.operator == "and":
                return a and b
            elif self.operator == "or":
                return a or b
            
            
        def __str__(self):
            return self.operator

        @staticmethod
        def is_operator(str: str, index: int):

            # match to operator
            operator = str[index] if str[index] in __class__.OPERATORS else None
            operator = str[index:index+2] if str[index:index+2] == "**" else operator

            # handle neagtive sign
            if operator == '-':
                if str[index] == ')' or str[index].isnumeric():
                    return operator
                else:
                    return None

            return operator

        @staticmethod
        def lowest_priority(operators: list[(int, str)]):
            highest = operators[0]

            for i, operator in operators:
                if __class__.OPERATOR_PRECEDENCE[operator] < __class__.OPERATOR_PRECEDENCE[highest[1]]:
                    highest = (i, operator)

            return highest

    class Term():
        """
        A term in an algebraic equation. This can be a constant or a variable.
        """
        def __init__(self, name: str):
            self.name = name

        def __str__(self):
            return self.name

    class Equation():
        """
        A line can be defined by an algebraic equation. This class represents that equation
        """

        def __init__(
            self,
            a: Union['Equation', float, 'Term'],
            b: Union['Equation', float, 'Term'],
            operator: 'Operator'
        ):
            self.a = a
            self.b = b
            self.operator = operator

        def __str__(self):
            a_str = f"({self.a})" if isinstance(self.a, __class__) else str(self.a)
            b_str = f"({self.b})" if isinstance(self.b, __class__) else str(self.b)

            return f"{a_str} {self.operator} {b_str}"

        def result(self, term_value: float | int):
            a = self.a.result(term_value) if isinstance(self.a, __class__) else self.a
            b = self.b.result(term_value) if isinstance(self.b, __class__) else self.b

            a = a if isinstance(a, float) else term_value
            b = b if isinstance(b, float) else term_value

            return self.operator.apply(a, b)

    @staticmethod
    def parse_equation(equation: str) -> Equation | float | Term:
        
        equation = equation.replace(' ', '')

        # replace 'rand' keyword with a random number
        i = 0
        while i < len(equation):
            if equation[i] == 'r' and len(equation) - i > 3 and equation[i:i+4] == 'rand':
                equation = equation[:i] + str(randint(0, 100)) + equation[i+4:]
            i += 1
        
        # find lowest priority operator not enclosed in parentheses
        parens = []
        unenclosed_operators = []
        skip = False
        for i, char in enumerate(equation):
            if skip:
                skip = False
                continue
            
            if char == '=':
                last_char = equation[i-1] if i > 0 else None
                next_char = equation[i+1] if i < len(equation) - 1 else None
                if last_char != '=' and next_char != '=':
                    raise Mip.MipException(
                        """
                        Equation shouldn't have a '=' sign. Convert all function to standard form first.

                        Example:
                        2 + y = 3x**2 + 4 -> 3x**2 + 4 - 2

                        x = ( (-2 + y) / 3 )**(1/2) is also a valid form of the equation above, which you 
                        should provide as -> ((-2 + y) / 3)**(1/2)
                        """
                    )

            if char == '(':
                parens.append((i, char))
            if char == ')':
                if i == len(equation) - 1:
                    if len(parens) == 1 and parens[0][0] == 0:
                        return __class__.parse_equation(equation[1:-1])
                    break
                parens.pop()

            operator = __class__.Operator.is_operator(equation, i)
            if operator != None and len(parens) == 0:
                unenclosed_operators.append((i, operator))
                skip = len(operator) == 2

        if len(unenclosed_operators) != 0:
            i, operator = __class__.Operator.lowest_priority(unenclosed_operators)
            a = __class__.parse_equation(equation[:i])
            b = __class__.parse_equation(equation[i+len(operator):])
            return __class__.Equation(a, b, __class__.Operator(operator))
            

        if equation.isnumeric():
            return float(equation)
        elif equation[0] == '-' and equation[1:].isnumeric():
            return float(equation)
        elif equation == 'x' or equation == 'y':
            return __class__.Term(equation)
        else:
            raise Mip.MipException("Invalid equation. Tried to convert this to a float and failed: " + equation)
    

    class Animate():
        """
        An animation that can be applied to lines, curves, fills, and pixels. It is defined
        by a list of key frames with a point, rotation, scale and paired with a time at which 
        it occurs. The repeat_count controls how many times the animation should be repeated. 
        Left blank it repeats forever but can also be replaced with an int

        Mip will interpolate between the key frames to create the animation.

        For animating fill operations, if the fill ran against lines, those lines are
        translated with the fill. Otherwise only the fill is translated.

        Syntax:
        - animate [x1 y1 (rot) (scale)] occurtime (repeat_count)

        Example:
        - animate [25 25] 0s [50 50] 10s [25 25] 20s
        - animate [0 0 0 0] 0s [0 0 0 1] 10s [0 0 0 0] 20s
        - animate [0 0 0 0] 0s [0 0 360] 10s
        - animate [0 0 0 0] 0s [0 0 360] 1.1s 1

        This first example translates the target from 25 25 to 50 50 in 10 seconds, then back to 25 25 in 10 seconds. This is repeated forever.
        The second example scales the target from 0 0 to 1 1 in 10 seconds, then back to 0 0 in 10 seconds. This is repeated forever
        The third example rotates the target 360 degrees in 10 seconds. This is repeated forever
        The fourth example rotates the target 360 degrees in 1 second, only one time.

        """
        def __init__(
            self,
            key_frames: list[(int, int, int, int), str | float], 
            repeat: str | int, 
            ranges: list[int]
        ):
            self.key_frames = key_frames
            self.repeat = repeat
            self.ranges = ranges

        """
        Applies the animation based off the keyframes and the current frame.
        Each frame is 1/120th of a second. So if you render every frame you'd 
        be rendering at 120fps. (That's be overkill though so you'll probably just 
        want to do ~40fps or so instead)
        """
        def animate(self, pixels, width, height, frame):
            seconds = frame / 120.0

            index = bisect.bisect(self.ranges, frame)
            i_0, i_1 = index, index + 1
            if i_1 >= len(self.key_frames): i_1 = 0

            x1, y1, rot1, scale1 = self.key_frames[i_0]
            x2, y2, rot2, scale2 = self.key_frames[i_1]


            # interpolate between keyframes
            interp_value = (self.ranges[i_0] + seconds) / self.ranges[i_1]
            x = x1 + (x2 - x1) * interp_value
            y = y1 + (y2 - y1) * interp_value
            rot = rot1 + (rot2 - rot1) * interp_value
            scale = scale1 + (scale2 - scale1) * interp_value

            # apply scale
            if scale != 1:
                pixels = self.__scale(pixels, scale, scale, width, height)

                # account for scaling offset
                x -= int(x / scale)
                y -= int(y / scale)

            # apply translation
            if x != 0 or y != 0:
                pixels = self.__translate(pixels, x, y, width, height)
                
            # apply rotation
            if rot != 0 and rot != 360:
                pixels = self.__rotate(pixels, rot, width, height)

            return pixels


        def __scale(self, pixels: list[(int, int, int, int)], xscale: float, yscale: float, width: int, height: int):
            # uses Bilinear Interpolation to scale the image

            # Calculate the new dimensions
            new_width = int(width * xscale)
            new_height = int(height * yscale)

            # Create an empty numpy array for the output image
            new_pixels = [None] * new_width * new_height

            # Perform the bilinear interpolation
            for x in range(new_width):
                for y in range(new_height):

                    # Map the pixel in the output image to its corresponding location in the input image
                    pos_x = x * xscale
                    pos_y = y * yscale
                    
                    src_x = int(pos_x)
                    src_y = int(pos_y)
                    if pixels[src_y * width + src_x] == None: continue

                    # Get two surrounding pixels
                    src_x1 = src_x + 1 if src_x < len(pixels) - 1 else src_x - 1
                    src_y1 = src_y
                    src_x2 = src_y
                    src_y2 = src_y + width if src_y < len(pixels) - width else src_y - width

                    # Calculate the weights for each pixel
                    source_pixels = [(src_x, src_y), (src_x1, src_y1), (src_x2, src_y2)]
                    weights = []
                    total_weight = 0
                    for x1, y1 in source_pixels:
                        weight = abs(pos_x - x1) + abs(pos_y - y1)
                        weights.append(weight)
                        total_weight += weight

                    # Normalize the weights
                    for i in range(len(weights)):
                        weights[i] /= total_weight

                    r, g, b, a = 0, 0, 0, 0
                    for i, x1, y1 in enumerate(source_pixels):
                        pixel = pixels[y1 * width + x1]
                        r += pixel[0] * weights[i]
                        g += pixel[1] * weights[i]
                        b += pixel[2] * weights[i]
                        a += pixel[3] * weights[i]

                    # Calculate the output pixel value
                    new_pixels[x + y * new_width] = (r, g, b, a)

            # Return the output image
            return new_pixels

        def __translate(self, pixels: list[(int, int, int, int)], x: int, y: int, width: int, height: int):
            new_pixels = [None] * len(pixels)

            for x_i in range(width):
                for y_i in range(height):
                    if x_i + x < width and y_i + y < height:
                        i = y_i * width + x_i
                        new_pixels[i + y * width + x] = pixels[i]

            return new_pixels

        def __rotate(self, pixels: list[(int, int, int, int)], rot: int, width: int, height: int):
            # determine origin of shape

            origin_x = 0
            origin_y = 0
            counted = 0
            for i in range(len(pixels)):
                if pixels != None:
                    origin_x += i % width
                    origin_y += i // width
                    counted += 1
            origin_x /= counted
            origin_y /= counted

            # rotate the image
            new_pixels = [None] * len(pixels)
            rot = rot * cmath.pi / 180
            cos_rot = cos(rot)
            sin_rot = sin(rot)
            for i, pixel in enumerate(pixels):
                if pixel != None:
                    x, y = i % width, i // width
                    new_x, new_y = x * cos_rot - y * sin_rot, x * sin_rot + y * cos_rot
                    new_pixels[int(new_y) * width + int(new_x)] = pixel

            return new_pixels


    class Line():
        """
        Line to be drawn. Lines can either be defined by two coordinates,
        or by an algebraic equation. 

        Syntax:
        - line x1 y1 x2 y2 (color) (thickness)
        - line 'equation' (color) (thickness) (x1-x2)

        Examples:
        - line 0 0 100 100
        - line '2x + 3' #1a1919 2
        - line '(2x**2)/3 + 3' #1a1919 2
        - line 'y * 2 + 2y + 77' 'rgba(255, 0, 0, 255)' 2
        """

        @classmethod
        def with_coordinates(
            cls,
            x1: int, 
            y1: int, 
            x2: int, 
            y2: int, 
            color: tuple[int, int, int, int], 
            thickness: int = 1
        ):
            cls.x1 = x1
            cls.y1 = y1
            cls.x2 = x2
            cls.y2 = y2
            cls.color = color
            cls.thickness = thickness

            return cls

        @classmethod
        def with_equation(
            cls,
            equation: 'Equation', 
            color: tuple[int, int, int, int], 
            thickness: int = 1,
            range_x_or_y: list[int] = None,
        ):
            cls.equation = equation
            cls.color = color
            cls.thickness = thickness
            cls.range_x_or_y = range_x_or_y

            return cls

        def draw(self, pixels, width, height):
            if self.equation:
                use_x = False if 'y' in self.equation else True
                var_range = self.range_x_or_y if self.range_x_or_y else range(
                    0, 
                    width if use_x else height
                )
                for i in range(var_range[0], var_range[1]):
                    x = i if use_x else self.equation.result(i)
                    y = i if not use_x else self.equation.result(i)
                    pixels[y * width + x] = self.color
                    for j in range(1, self.thickness):
                        pixels[y * width + x + j] = self.color
                        pixels[(y + j) * width + x] = self.color
                        pixels[(y + j) * width + x + j] = self.color

            else:
                # Bresenham's line algorithm
                dx = abs(self.x2 - self.x1)
                dy = abs(self.y2 - self.y1)
                sx = 1 if self.x1 < self.x2 else -1
                sy = 1 if self.y1 < self.y2 else -1
                err = dx - dy
                x = self.x1
                y = self.y1

                while True:
                    pixels[y * width + x] = self.color
                    for i in range(1, self.thickness):
                        pixels[y * width + x + i] = self.color
                        pixels[(y + i) * width + x] = self.color
                        pixels[(y + i) * width + x + i] = self.color

                    if x == self.x2 and y == self.y2:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err = err - dy
                        x = x + sx
                    if e2 < dx:
                        err = err + dx
                        y = y + sy

    class Curve():
        """
        A curve is a quadratic bézeir curve defined by 2 endpoints and another point
        that controls the curve

        Syntax:
        - curve x1 y1 x2 y2 control_point_x control_point_y (color) (thickness)
        """
        def __init__(
            self,
            x1: int, 
            y1: int, 
            x2: int, 
            y2: int,
            control_point_x: int, 
            control_point_y: int, 
            color: tuple[int, int, int, int], 
            thickness: int = 1      
        ):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.control_point_x = control_point_x
            self.control_point_y = control_point_y
            self.color = color
            self.thickness = thickness


        def draw(self, pixels, width, height):
            # Bézier curve algorithm
            for t in range(0, 100):
                x = (1-t)**2 * self.x1 + 2 * (1-t) * t * self.control_point_x + t**2 * self.x2
                y = (1-t)**2 * self.y1 + 2 * (1-t) * t * self.control_point_y + t**2 * self.y2
                pixels[y * width + x] = self.color
                for i in range(1, self.thickness):
                    pixels[y * width + x + i] = self.color
                    pixels[(y + i) * width + x] = self.color
                    pixels[(y + i) * width + x + i] = self.color

    class Fill():
        """
        Fill starts at the provided point filling surrounding pixels of the same
        color as the starting point

        Syntax:
        - fill x1 y1 (color)
        """
        def __init__(
            self,
            x1: int, 
            y1: int, 
            color: tuple[int, int, int, int], 
        ):
            self.x1 = x1
            self.y1 = y1
            self.color = color

        def draw(self, pixels, width, height):
            color_to_replace = pixels[self.y1 * width + self.x1]
            stack = [(self.x1, self.y1)]
            while len(stack) > 0:
                x, y = stack.pop()
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                if pixels[y * width + x] == color_to_replace:
                    pixels[y * width + x] = self.color
                    stack.append((x + 1, y))
                    stack.append((x - 1, y))
                    stack.append((x, y + 1))
                    stack.append((x, y - 1))
        
    class Pixel():
        """
        Pixel is a way to set the color of a certain pixel. It can also be given
        a conditional expression as a way of setting pixels based off their x y
        coordinates.

        Syntax:
        - pixel x1 y1 (color)
        - pixel 'conditional_expression' (color)

        valid operators: x, y, ==, !=, <, >, %, /, *, **, -, +, and, or, rand

        Examples:
        - pixel 0 0 #1a1919
        - pixel 'x % 2 == 0 and y % 2 == 0' #1a1919
        - pixel 'x > 10 and x < 100 and y > 10 and y < 100' rgba(255, 0, 0, 255)

        """


        class Operator():
            """
            An operator in an algebraic equation. This can be +, -, *, / or **.
            """
            
            OPERATORS = {'==', '!=', '<', '>', 'and', 'or', '+', '-', '*', '/', '%', '**'}
            OPERATOR_PRECEDENCE = {
                '+': 1,
                '-': 1,
                '*': 2,
                '/': 2,
                '%': 2,
                '**': 3
            }

            def __init__(
                self,
                operator: str
            ):
                self.operator = operator
            
            def apply(self, a: float, b: float):
                if self.operator == "+":
                    return a + b
                elif self.operator == "-":
                    return a - b
                elif self.operator == "*":
                    return a * b
                elif self.operator == "/":
                    return a / b
                elif self.operator == "**":
                    return a ** b
                elif self.operator == "%":
                    return a % b
                elif self.operator == "==":
                    return a == b
                elif self.operator == "!=":
                    return a != b
                elif self.operator == "<":
                    return a < b
                elif self.operator == ">":
                    return a > b
                elif self.operator == "and":
                    return a and b
                elif self.operator == "or":
                    return a or b
                
            def __str__(self):
                return self.operator

            @staticmethod
            def is_operator(str: str, index: int):

                # match to operator
                o = str[index]
                op = str[index:index+2] if len(str) > index + 1 else None
                opp = str[index:index+3] if len(str) > index + 2 else None
                operator = o if o in __class__.OPERATORS else None
                operator = str[index:index+2] if str[index:index+2] == "**" else operator

                # handle neagtive sign
                if operator == '-':
                    if str[index] == ')' or str[index].isnumeric():
                        return operator
                    else:
                        return None

                return operator

            @staticmethod
            def lowest_priority(operators: list[(int, str)]):
                highest = operators[0]

                for i, operator in operators:
                    if __class__.OPERATOR_PRECEDENCE[operator] < __class__.OPERATOR_PRECEDENCE[highest[1]]:
                        highest = (i, operator)

                return highest


        class ConditionalExpression():
            def __init__(
                self,
                left: Union['ConditionalExpression', float, str],
                operator: str,
                right: Union['ConditionalExpression', float, str]
            ):
                self.expression = expression

            def result(self, x: int, y: int):
                return eval(self.expression)


        def __init__(
            self,
            x1: int, 
            y1: int, 
            color: tuple[int, int, int, int], 
        ):
            self.x1 = x1
            self.y1 = y1
            self.color = color
        

        def __init__(
            self,
            conditional_expression: str,
            color: tuple[int, int, int, int]    
        ):
            self.conditional_expression = Mip.parse_equation(conditional_expression)
            self.color = color


        def draw(self, pixels, width, height):
            if self.conditional_expression:
                for x in range(width):
                    for y in range(height):
                        if self.conditional_expression.result(x, y):
                            pixels[y * width + x] = self.color
            else:
                pixels[self.y1 * width + self.x1] = self.color
        

    class MipException(Exception):
        def __init__(self, message):
            super().__init__(message)



    def __init__(
            self, 
            width : int, 
            height : int, 
            color : str,
            operations: list[Line | Curve | Fill | Pixel | Animate]
        ):
        self.width = width
        self.height = height
        self.color = __class__.__parse_color(color)
        self.is_drawn = False

        self.operations = operations
        self.background_pixels = []
        self.first_animation = None


    def parse(
        self,
        str: str
    ):
        for i, line in enumerate(str.split['\n']):
            if line.startswith("mip"):
                width, height, color = self.__parse_mip(line)
                self.width = width
                self.height = height
                self.color = color
            elif line.startswith("line"):
                self.operations.append(self.__parse_line(line))
            elif line.startswith("curve"):
                self.operations.append(self.__parse_curve(line))
            elif line.startswith("fill"):
                self.operations.append(self.__parse_fill(line))
            elif line.startswith("pixel"):
                self.operations.append(self.__parse_pixel(line))
            elif line.startswith("animate"):
                self.operations.append(self.__parse_animate(line))
        
        self.pixels = []
            
    def __parse_mip(self, line: str):
        
        tokens = line.split(" ")
        if tokens[0] != "mip":
            raise Mip.MipException("file must start with 'mip' command")
        
        if len(tokens) < 3:
            raise Mip.MipException("mip command must have at least 2 arguments: width, height. Color is an optional third argument")
        
        width = tokens[1]
        height = tokens[2]
        color = tokens[3] if len(tokens) == 4 else Mip.DEFAULT_BACKGROUND

        return width, height, color

    def __parse_line(self, line: str) -> Line:
        """
        Syntax:
        - line x1 y1 x2 y2 (color) (thickness)
        - line 'equation' (color) (thickness)
        """

        # handle an algebraic expression
        if '\'' in line:
            first_quote = line.index()
            second_quote = line[first_quote+1:].index()
            equation = __class__.Line.parse_equation( line[first_quote:second_quote] )
            line = line[:first_quote] + line[second_quote:]
            tokens = line.split(" ")
            color = tokens[0] if len(tokens) > 0 else Mip.DEFAULT_COLOR
            thickness = tokens[1] if len(tokens) > 1 else 1

            range_x_or_y = None
            if len(tokens) > 2:
                str_range = tokens[2]
                x1, x2 = str_range.split('-')
                x1, x2 = float(x1), float(x2)
                range_x_or_y = [x1, x2]

            return __class__.Line().with_equation(equation, color, thickness, range_x_or_y)


        # otherwise handle the typical line
        tokens = line.split(" ")
        if len(tokens) < 5:
            # XXX do algebraic equations here
            raise Mip.MipException("line command must have 4 arguments: x1, y1, x2, y2, color. Color and thickness are other optional arguments")

        x1 = int(tokens[1])
        y1 = int(tokens[2])
        x2 = int(tokens[3])
        y2 = int(tokens[4])
        color = tokens[5] if len(tokens) > 5 else Mip.DEFAULT_COLOR
        thickness = int(tokens[6]) if len(tokens) == 7 else 1

        return Mip.Line(x1, y1, x2, y2, color, thickness)

    def __parse_curve(self, line: str) -> Curve:
        """
        Syntax:
        - curve x1 y1 x2 y2 control_point_x control_point_y (color) (thickness)
        """
        tokens = line.split(" ")
        if len(tokens) < 7:
            raise Mip.MipException("curve command must have 6 arguments: x1, y1, x2, y2, control_point_x, control_point_y. Color and  Thickness are optional")
        
        x1 = int(tokens[1])
        y1 = int(tokens[2])
        x2 = int(tokens[3])
        y2 = int(tokens[4])
        control_point_x = int(tokens[5])
        control_point_y = int(tokens[6])

        color = tokens[7] if len(tokens) > 7 else Mip.DEFAULT_COLOR
        thickness = int(tokens[8]) if len(tokens) == 9 else 1

        return Mip.Curve(x1, y1, x2, y2, control_point_x, control_point_y, color, thickness)

    def __parse_fill(self, line: str) -> Fill:
        """
        Syntax:
        - fill x1 y1 (color)
        """
        tokens = line.split(" ")
        if len(tokens) < 3:
            raise Mip.MipException("fill command must have 2 arguments: x1, y1. Color is optional")

        x1 = int(tokens[1])
        y1 = int(tokens[2])
        color = tokens[3] if len(tokens) > 2 else Mip.DEFAULT_COLOR

        return Mip.Fill(x1, y1, color)

    def __parse_pixel(self, line: str) -> Pixel:
        """
        Syntax:
        - pixel x1 y1 (color)
        - pixel 'conditional_expression' (color)
        """
        tokens = line.split(" ")
        if len(tokens) < 3:
            raise Mip.MipException("pixel command must have 2 arguments: x1, y1. Color is optional")

        if tokens[1].startswith('\''):
            conditional_expression = tokens[1]
            color = tokens[2]
            return Mip.Pixel(conditional_expression, color)

        x1 = int(tokens[1])
        y1 = int(tokens[2])
        color = tokens[3]

        return Mip.Pixel(x1, y1, color)

    def __parse_animate(self, line: str) -> Animate:
        """
        Syntax:
        - animate [x1 y1 (rot) (scale)] (time) ... (repeat_count)
        """
        key_frames_strs = line.split("[")

        key_frames = []
        ranges = []
        repeat = 'forever'
        for i in range(1, len(key_frames_strs)):
            key_frame_str = key_frames_strs[i]
            brac_index = key_frame_str.index(']')
            
            # parse key frame
            params_str = key_frame_str[:brac_index]
            tokens = params_str.split(" ")
            x1 = int(tokens[0])
            y1 = int(tokens[1])
            rot = int(tokens[2]) if len(tokens) > 2 else 0
            scale = int(tokens[3]) if len(tokens) > 3 else 1
            
            # parse time
            time = key_frame_str[brac_index+1:]
            time.replace('', '')
            tokens = time.split('s')
            time = int(tokens[0])
            if i == len(key_frames_strs) - 1 and len(tokens) > 1: 
                repeat = int(tokens[1])
            
            key_frames.append((
                (x1, y1, rot, scale), 
                time
            ))

            # add to time ranges
            ranges.append(time)

        return Mip.Animate(key_frames, repeat, ranges)
            


            


        return Mip.Animate(x1, y1, color)


    def draw(self):
        self.background_pixels = []
        for _ in range(self.width):
            for _ in range(self.height):
                self.background_pixels.append(self.color)
        
        # set animations on operations
        set_operations = []
        for operation in self.operations:
            if isinstance(operation, __class__.Animate):
                set_operations[-1].animation = operation
            else:
                set_operations.append(operation)

        self.operations = set_operations


        doing_background = True
        pixels = []
        for i, operation in enumerate(self.operations):
            if doing_background and operation.animation == None:
                operation.draw(self.background_pixels, self.width, self.height)
            elif doing_background and operation.animation != None:
                doing_background = False
                self.first_animation = i
                pixels = self.background_pixels[:]
                operation.draw(pixels, self.width, self.height)
            elif not doing_background:
                operation.draw(pixels, self.width, self.height)

        return pixels


    def animate(self, frame: int = 0):
        if self.background_pixels.empty():
            self.draw()

        pixels = self.background_pixels[:]
        for i in range(self.first_animation, len(self.operations)):
            operation = self.operations[i]
            op_pixels = operation.draw([None] * len(pixels) , self.width, self.height)
            if operation.animation != None:
                op_pixels = operation.animation.animate(pixels, self.width, self.height, frame)
            
            for i, pixel in enumerate(op_pixels):
                if pixel != None:
                    pixels[i] = pixel
                
        
        return pixels

        


    @staticmethod
    def __parse_color(color: str) -> tuple[int, int, int, int]:
        """
        Parses a color string and returns an RGBA tuple.

        Parameters:
        color (str): The color as a string.

        Returns:
        tuple: The color as an RGBA tuple.
        """
        if color.startswith("rgba"):
            color = color[5:-1]
            color = re.sub(r'[()\s]', '', color)
            color = color.split(",")
            r, g, b, a = int(color[0]), int(color[1]), int(color[2]), int(color[3])
        else:
            r, g, b, a = __class__.__hex_to_rgba(color)
            
        return r, g, b, a

    @staticmethod
    def __hex_to_rgba(hex_color):
        """
        Converts a hex color to an RGBA tuple.

        Parameters:
        hex_color (str): The color as a hex string (e.g., "#FFFFFF" or "#FFFFFFFF").

        Returns:
        tuple: The color as an RGBA tuple (e.g., (255, 255, 255, 255)).
        """
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            a = 255
        elif len(hex_color) == 8:
            r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
        else:
            raise ValueError("Invalid hex color - must be 6 or 8 characters (not including '#')")
        return r, g, b, a