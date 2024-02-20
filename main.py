from PIL import Image, ImageDraw, ImageChops
from evol import Population
import random

SIDES = 5  # Amount of corners in each Polygon
POLYGON_COUNT = 100  # Amount of Polygons generated
# Constants for image
MAX = 255 * 200 * 200
TARGET = Image.open("Images/3a.png")
TARGET.load()  # read image and close the file


# Make the polygon and randomise the coordinates of the polygon corners
def make_polygon(n):
    R, G, B = [random.randrange(256) for i in range(3)]
    A = random.randrange(30, 60)
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = [random.randrange(10, 190) for i in range(10)]

    # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190
    return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]


def initialise():
    return [make_polygon(SIDES) for i in range(POLYGON_COUNT)]


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


def evaluate(solution):
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


population = Population.generate(initialise, evaluate, size=10, maximize=True)
draw(population[0].chromosome).save("solution.png")
