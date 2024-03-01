from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution
import random

SIDES = 3
POLYGON_COUNT = 100
# Constants for image
MAX = 255 * 200 * 200
TARGET = Image.open("Images/3a.png")
TARGET.load()  # read image and close the file


# Make the polygon and randomise the coordinates of the polygon corners
def make_polygon(n):
    R, G, B = [random.randrange(256) for i in range(3)]
    A = random.randrange(30, 60)
    x1, y1, x2, y2, x3, y3 = [random.randrange(10, 190) for i in range(6)]

    # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190
    return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3)]


def initialise():
    return [make_polygon(SIDES) for i in range(POLYGON_COUNT)]


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


def mutate(solution, rate):
    solution = list(solution)

    if random.random() < 0.3:  # Should be a value between 0 and 1 (percentage)
        # mutate points
        i = random.randrange(len(solution))
        polygon = list(solution[i])
        coords = [x for point in polygon[1:] for x in
                  point]  # Make a flat list of all the x and y coordinates the points
        coords = [x if random.random() > rate else  # For each coordinate, sometimes randomly change it a little bit
                  x + random.normalvariate(0, 10) for x in coords]
        coords = [max(0, min(int(x), 200)) for x in coords]  # Make sure the polygon stays within the boundaries
        polygon[1:] = list(zip(coords[::2], coords[1::2]))  # Put these adjusted coordinates back into pairs
        solution[i] = polygon
    else:
        # reorder polygons
        random.shuffle(solution)

    return solution


def select(population):
    return [random.choice(population) for i in range(2)]


def combine(*parents):
    return [a if random.random() < 0.5 else b for a, b in zip(*parents)]


def evaluate(solution):
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


def evolve(population, args):
    population.survive(fraction=0.5)
    population.breed(parent_picker=select, combiner=combine)
    population.mutate(mutate_function=mutate, rate=0.1)
    return population


population = Population.generate(initialise, evaluate, size=10, maximize=True)
for i in range(6000):
    evolve(population, evaluate)
    print(evaluate(solution=population[0].chromosome))

draw(population[0].chromosome).save("solution.png")
