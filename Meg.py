from abc import ABC, abstractmethod
import bisect
from cmath import cos, sin
import cmath
from math import floor
from random import randint
import re
import struct
import threading
import time
from typing import Union
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene
from PySide6.QtCore import Qt, QRectF, QTimer
import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtWidgets
import sys
import numpy as np

class Meg():
    def __init__(self, **args):
        self.app = QApplication(sys.argv)
        self.window = MegWindow(**args)
        sys.exit(self.app.exec_())


DEFAULT_COLOR = "#1a1919"

class MegWindow(QMainWindow):
    def __init__(
            self, 
            width=None, 
            height=None, 
            rounded_corners=False,
            screen_x=0,
            screen_y=0
        ):
        super().__init__()

        self.rounded_corners = rounded_corners

        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()

        
        # set needed variables
        self.width = screen_geometry.width() if width == None else width
        self.height = screen_geometry.height() if height == None else height


        # remove os frame and make the window transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Set the window size
        print(self.width, self.height)
        self.setGeometry(screen_x, screen_y, self.width, self.height)

        self.mpos = None

        # make worker to update image
        t = time.time()
        self.lu_image = __class__.make_lu('test.sh')
        print("Time to make image", time.time() - t)
        threading.Thread(target=self.update_image_worker).start()

        # Show the window
        self.show()


    @staticmethod
    def make_lu(path: str):

        # read the file
        file_contents = None
        with open(path, 'r') as file:
            # Read the entire contents of the file
            file_contents = file.read()

        # draw lu image
        lu = Lu.parse(file_contents)
        pixels = lu.draw()
        
        # Convert pixels to bytes to image
        pixels_bytes = pixels.tobytes()
        image = QtGui.QImage(pixels_bytes, lu.width, lu.height, QtGui.QImage.Format_RGBA8888)
        
        return image

    def update_image_worker(self):
        while True:
            self.lu_image = __class__.make_lu('test.sh')
            print("Updated image", time.time())
            self.update()
            time.sleep(1)

    # mouse events
    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        print(f"Mouse pressed at ({x}, {y})")
        self.mpos = event.pos()

    def mouseMoveEvent(self, event):
        if self.mpos is not None and event.buttons() == Qt.LeftButton:
            diff = event.pos() - self.mpos
            new_pos = self.pos() + diff
            self.move(new_pos)


    # method called by QT to paint the window
    def paintEvent(self, event):

        print("Repaint", time.time())

        # make the window background with rounded corners
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtGui.QColor(DEFAULT_COLOR))  # RGBA color
        painter.setPen(QtGui.QPen(QtGui.QColor(DEFAULT_COLOR), 0))
        path = QtGui.QPainterPath()
        radius = 10

        if self.rounded_corners:
            path.addRoundedRect(QRectF(0, 0, self.width, self.height), radius, radius)
        else:
            path.addRect(QRectF(0, 0, self.width, self.height))

        painter.drawPath(path)

        # draw lu image
        if self.lu_image is not None:
            painter.drawImage(0, 0, self.lu_image)
        painter.end()



#############################
### Lu Image Format Code ###
#############################

class Operator():
    """
    An operator in an algebraic equation or conditional expression. This can be +, -, *, /, **; for
    algebra, and ==, !=, <, >, and, or; for conditional expressions.
    """
    
    OPERATORS = {'+', '-', '*', '/', '**', '==', '!=', '<', '>', 'and', 'or'}
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
                raise Lu.LuException(
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
        raise Lu.LuException("Invalid equation. Tried to convert this to a float and failed: " + equation)

@staticmethod
def plot_point(
    pixels, 
    width, 
    height, 
    x, 
    y, 
    brush_shape, 
    color
):
    # if 0 <= x < width and 0 <= y < height:
    #     pixels[y * width + x] = color
    #     for i in range(0, brush_shape):
    #         for j in range(0, brush_shape):
    #             y_j = y + j
    #             x_i = x + i
    #             if 0 <= x_i < width and 0 <= y_j < height:
    #                 pixels[y_j * width + x_i] = color

    if 0 <= x < width and 0 <= y < height:
        pixels[y, x] = color
        for i, j in brush_shape:
            y_j = y + j
            x_i = x + i
            if 0 <= x_i < width and 0 <= y_j < height:
                pixels[y_j, x_i] = color

@staticmethod
def make_circle_brush(radius: int):
    circle = []
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            if i * i + j * j <= radius * radius:
                circle.append((i, j))
    return circle


class Animate():
    """
    An animation that can be applied to lines, curves, fills, and pixels. It is defined
    by a list of key frames with a point, rotation, scale and paired with a time at which 
    it occurs. The repeat_count controls how many times the animation should be repeated. 
    Left blank it repeats forever but can also be replaced with an int

    Lu will interpolate between the key frames to create the animation.

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

    def __init__(
        self,
        x1: int = None, 
        y1: int = None,
        x2: int = None, 
        y2: int = None, 
        equation: Equation = None, 
        color: np.ndarray = np.array([0, 0, 0, 255]),
        thickness: int = 1,
        range_x_or_y: list[int] = None
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.equation = equation
        self.color = color
        self.thickness = thickness
        self.range_x_or_y = range_x_or_y

        self.animation = None


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
                plot_point(pixels, width, height, x, y, make_circle_brush(self.thickness), self.color)


        else:
            # Bresenham's line algorithm
            dx = abs(self.x2 - self.x1)
            dy = abs(self.y2 - self.y1)
            sx = 0 if dx == 0 else (1 if self.x1 < self.x2 else -1)
            sy = 0 if dy == 0 else (1 if self.y1 < self.y2 else -1)
            err = dx - dy
            x = self.x1
            y = self.y1

            while True:
                if (x < 0 or x >= width) or (y < 0 or y >= height) or (sx == 1 and x > self.x2) or (sy == 1 and y > self.y2) or (sx == -1 and x < self.x2) or (sy == -1 and y < self.y2):
                    break

                plot_point(pixels, width, height, x, y, make_circle_brush(self.thickness), self.color)

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
        color: np.ndarray = np.array([0, 0, 0, 255]), 
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
        self.animation = None


    def draw(self, pixels, width, height):



        step = 1.0 / max(width, height)
        t = 0
        while t < 1.0:
            # a = q + (1-t)(p-q)
            # b = q + t(s-q) 
            # final = a + t(b-a)
            # p = xy1, s = xy2, q = control_point
            ax = self.control_point_x + (1-t) * (self.x1 - self.control_point_x)
            bx = self.control_point_x + t * (self.x2 - self.control_point_x)
            x = int(ax + t * (bx - ax))
            ay = self.control_point_y + (1-t) * (self.y1 - self.control_point_y)
            by = self.control_point_y + t * (self.y2 - self.control_point_y)
            y = int(ay + t * (by - ay))
            plot_point(pixels, width, height, x, y, make_circle_brush(self.thickness), self.color)

            t += step
         
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
        color: np.ndarray = np.array([0, 0, 0, 255]), 
    ):
        self.x1 = x1
        self.y1 = y1
        self.color = color
        self.animation = None

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
        x1: int = None, 
        y1: int = None, 
        conditional_expression: str = None,
        color: np.ndarray = np.array([0, 0, 0, 255]), 
    ):
        self.x1 = x1
        self.y1 = y1
        if conditional_expression:
            self.conditional_expression = Lu.parse_equation(conditional_expression)
        self.color = color
        self.animation = None


    def draw(self, pixels, width, height):
        if self.conditional_expression:
            for x in range(width):
                for y in range(height):
                    if self.conditional_expression.result(x, y):
                        pixels[y * width + x] = self.color
        else:
            pixels[self.y1 * width + self.x1] = self.color
    
class Lu(ABC):
    """
    Lu is a image format inspired by SVG. This class parses the Lu format and stores the attributes
    internally. These attributes can then be converted to an image with the draw method which produces
    an array of pixel values. 

    Attributes:
        width (int): The width of the Lu.
        height (int): The height of the Lu.
        color (str): The color of the Lu.
    """

    DEFAULT_BACKGROUND = 'rgba(0, 0, 0, 0)'
    DEFAULT_COLOR = 'rgba(26, 25, 25, 255)'



    class LuException(Exception):
        def __init__(self, message):
            super().__init__(message)



    def __init__(
            self, 
            width : int, 
            height : int, 
            color : np.array = np.array([0, 0, 0, 0]),
            operations: list[Line | Curve | Fill | Pixel | Animate] = None
        ):
        self.width = int(width)
        self.height = int(height)
        self.color = color
        self.is_drawn = False

        self.operations = operations if operations else []
        self.background_pixels = np.zeros((self.width, self.height, 4), dtype=np.uint8)
        self.first_animation = None

    @staticmethod
    def parse(
        str: str
    ):
        lu = None
        for i, line in enumerate(str.split('\n')):
            line = line.replace(', ', ',')
            if line.startswith("lu"):
                width, height, color = __class__.__parse_lu(line)
                lu = Lu(width, height, color)
            elif line.startswith("line"):
                lu.operations.append(__class__.__parse_line(line))
            elif line.startswith("curve"):
                lu.operations.append(__class__.__parse_curve(line))
            elif line.startswith("fill"):
                lu.operations.append(__class__.__parse_fill(line))
            elif line.startswith("pixel"):
                lu.operations.append(__class__.__parse_pixel(line))
            elif line.startswith("animate"):
                lu.operations.append(__class__.__parse_animate(line))
            elif line.startswith("#"):
                continue
            elif len(line) > 0:
                raise Lu.LuException(f"Invalid command at line {i}: {line.split(' ')[0]}")

        return lu
           
    @staticmethod 
    def __parse_lu(line: str):
        """
        Syntax:
        - lu width height (color)
        """
        
        tokens = line.split(" ")
        if tokens[0] != "lu":
            raise Lu.LuException("file must start with 'lu' command")
        
        if len(tokens) < 3:
            raise Lu.LuException("lu command must have at least 2 arguments: width, height. Color is an optional third argument")
        
        width = int(tokens[1])
        height = int(tokens[2])
        color = tokens[3] if len(tokens) == 4 else Lu.DEFAULT_BACKGROUND
        color = __class__.parse_color(color)

        return width, height, color

    @staticmethod
    def __parse_line(line: str) -> Line:
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
            color = tokens[0] if len(tokens) > 0 else Lu.DEFAULT_COLOR
            color = __class__.parse_color(color)
            thickness = tokens[1] if len(tokens) > 1 else 1

            range_x_or_y = None
            if len(tokens) > 2:
                str_range = tokens[2]
                x1, x2 = str_range.split('-')
                x1, x2 = float(x1), float(x2)
                range_x_or_y = [x1, x2]

            return Line(equation=equation, color=color, thickness=thickness, range_x_or_y=range_x_or_y)


        # otherwise handle the typical line
        tokens = line.split(" ")
        if len(tokens) < 5:
            # XXX do algebraic equations here
            raise Lu.LuException("line command must have 4 arguments: x1, y1, x2, y2, color. Color and thickness are other optional arguments")

        x1 = int(tokens[1])
        y1 = int(tokens[2])
        x2 = int(tokens[3])
        y2 = int(tokens[4])
        color = tokens[5] if len(tokens) > 5 else Lu.DEFAULT_COLOR
        color = __class__.parse_color(color)
        thickness = int(tokens[6]) if len(tokens) == 7 else 1

        return Line(x1=x1, y1=y1, x2=x2, y2=y2, color=color, thickness=thickness)

    @staticmethod
    def __parse_curve(line: str) -> Curve:
        """
        Syntax:
        - curve x1 y1 x2 y2 control_point_x control_point_y (color) (thickness)
        """
        tokens = line.split(" ")
        if len(tokens) < 7:
            raise Lu.LuException("curve command must have 6 arguments: x1, y1, x2, y2, control_point_x, control_point_y. Color and  Thickness are optional")
        
        x1 = int(tokens[1])
        y1 = int(tokens[2])
        x2 = int(tokens[3])
        y2 = int(tokens[4])
        control_point_x = int(tokens[5])
        control_point_y = int(tokens[6])

        color = tokens[7] if len(tokens) > 7 else Lu.DEFAULT_COLOR
        color = __class__.parse_color(color)
        thickness = int(tokens[8]) if len(tokens) == 9 else 1

        return Curve(x1, y1, x2, y2, control_point_x, control_point_y, color, thickness)

    @staticmethod
    def __parse_fill(line: str) -> Fill:
        """
        Syntax:
        - fill x1 y1 (color)
        """
        tokens = line.split(" ")
        if len(tokens) < 3:
            raise Lu.LuException("fill command must have 2 arguments: x1, y1. Color is optional")

        x1 = int(tokens[1])
        y1 = int(tokens[2])
        color = tokens[3] if len(tokens) > 2 else Lu.DEFAULT_COLOR
        color = __class__.parse_color(color)

        return Fill(x1, y1, color)

    @staticmethod
    def __parse_pixel(line: str) -> Pixel:
        """
        Syntax:
        - pixel x1 y1 (color)
        - pixel 'conditional_expression' (color)
        """
        tokens = line.split(" ")
        if len(tokens) < 3:
            raise Lu.LuException("pixel command must have 2 arguments: x1, y1. Color is optional")

        if tokens[1].startswith('\''):
            conditional_expression = tokens[1]
            color = tokens[2]
            color = __class__.parse_color(color)
            return Pixel(conditional_expression, color)

        x1 = int(tokens[1])
        y1 = int(tokens[2])
        color = tokens[3]
        color = __class__.parse_color(color)

        return Pixel(x1, y1, color)

    @staticmethod
    def __parse_animate(line: str) -> Animate:
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

        return Animate(key_frames, repeat, ranges)
            


    def draw(self) -> np.ndarray:
        self.background_pixels = np.zeros((self.width, self.height, 4), dtype=np.uint8)
        color = np.array(self.color, dtype=np.uint8)
        for x in range(self.width):
            for y in range(self.height):
                self.background_pixels[x, y] = color
        
        # set animations on operations
        set_operations = []
        for operation in self.operations:
            if isinstance(operation, Animate):
                set_operations[-1].animation = operation
            else:
                set_operations.append(operation)

        self.operations = set_operations


        doing_background = True
        pixels = None
        for i, operation in enumerate(self.operations):
            if doing_background and operation.animation == None:
                operation.draw(self.background_pixels, self.width, self.height)
            elif doing_background and operation.animation != None:
                doing_background = False
                self.first_animation = i
                pixels = np.copy(self.background_pixels)
                operation.draw(pixels, self.width, self.height)
            elif not doing_background:
                operation.draw(pixels, self.width, self.height)

        if not pixels:
            pixels = np.copy(self.background_pixels)
        
        return pixels


    def animate(self, frame: int = 0) -> list[(int, int, int, int)]:
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
    def parse_color(color: str) -> np.ndarray:
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
            r, g, b, a = int(color[0]), int(color[1]), int(color[2]), float(color[3])
            a = int(a * 255) if a <= 1 else int(a)
        else:
            r, g, b, a = __class__.__hex_to_rgba(color)
            
        return np.array([r, g, b, a], dtype=np.uint8)

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