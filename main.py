import evol
from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution
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
        coords = [x for point in polygon[1:] for x in point]
        coords = [x if random.random() > rate else
                  x + random.normalvariate(0, 10) for x in coords]
        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))
        solution[i] = polygon
    else:
        # reorder polygons
        random.shuffle(solution)

    return solution


def evaluate(solution):
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


def func_to_optimise(xy):
    """
    This is the function we want to optimise (maximize)
    """
    x, y = xy
    return -(1 - x) ** 2 - (2 - y ** 2) ** 2


population = Population.generate(initialise, evaluate, size=10, maximize=True)


def pick_random_parents(population):
    """
    This is how we are going to select parents from the population
    """
    mom = random.choice(population)
    dad = random.choice(population)
    return mom, dad


def make_child(mom, dad):
    # Assuming you want to mix polygons from both parents to create a child
    # For simplicity, let's just concatenate half from each parent
    half = len(mom) // 2
    child = mom[:half] + dad[half:]
    return child


for i in range(100):
    population = (population
                  .survive(fraction=1)
                  .breed(parent_picker=pick_random_parents, combiner=make_child)
                  .mutate(mutate_function=mutate, rate=0.3))

    evo1 = (Evolution()
            .survive(fraction=0.5)
            .breed(parent_picker=pick_random_parents, combiner=make_child)
            .mutate(mutate_function=mutate, rate=1))

    evo2 = (Evolution()
            .survive(n=1)
            .breed(parent_picker=pick_random_parents, combiner=make_child)
            .mutate(mutate_function=mutate, rate=0.2))

    evo3 = (Evolution()
            .repeat(evo1, n=50)
            .repeat(evo2, n=10)
            .evaluate())

    population = population.evolve(evo3, n=5)
    print(f"the best score found: {max([i.fitness for i in population])}")


def run():
    draw(population[0].chromosome).save("solution.png")


run()
