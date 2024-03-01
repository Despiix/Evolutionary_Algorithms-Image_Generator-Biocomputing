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
    """
    Randomly assigns the colour, opaqueness and coordinates of the polygons in this instance
     I chose to use triangles because I believe they will be an easier shape to work with
    """
    R, G, B = [random.randrange(256) for i in range(3)]  # Assigns random int values to Red, Green and Blue
    A = random.randrange(30, 60)  # Sets random opaqueness
    x1, y1, x2, y2, x3, y3 = [random.randrange(10, 190) for i in range(6)]  # Sets random coordinates to polygons

    # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190
    return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3)]


def initialise():
    """Assigns the sides of the polygon and creates as many polygons as stated in the constants above"""
    return [make_polygon(SIDES) for i in range(POLYGON_COUNT)]


def draw(solution):
    """
    Parameters: Solution - which contains the 100 polygons generated in initialise().
    Create a blank picture: Make a new, empty picture that's colored and 200x200 pixels big.
    Draw Shapes: using the images colour and outline
    Returns: The picture with all the shapes drawn on it."""
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:  # For each shape in a list, draw it on the picture using its specific color and outline.
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


def mutate(solution, rate):
    """
    Mutates a given solution of polygons by either adjusting their coordinates slightly or by shuffling the order of polygons.
    The mutation is randomly applied  with a 50% chance, it modifies the coordinates of a randomly selected polygon.
    If a point's coordinate is selected for mutation, it keeps it within the boundaries.

    Alternatively, with a 50% chance, the function shuffles the order of all polygons in the solution.

    Parameters:
    - solution (list of lists): A list where each element represents a polygon.
    - rate (float): The mutation rate
    Returns:
    - list: The mutated solution, which may have one polygon's coordinates slightly adjusted or the order of polygons shuffled.
    """
    solution = list(solution)

    if random.random() < 0.5:  # Should be a value between 0 and 1 (percentage)
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
    """Select population randomly, chooses 2 parents"""
    return [random.choice(population) for i in range(2)]


def combine(*parents):
    """Combines the parents chosen in select to create a child"""
    return [a if random.random() < 0.5 else b for a, b in zip(*parents)]


def evaluate(solution):
    """
    Parameter:
        Solution - Generated polygons

    1. Compares two pictures to see where they don't match.
    2. Turns the differences into a simple black and white picture (That is what the "L" stands for).
    3. Calculates how much the pictures don't match by adding up the differences.
    4. The closer the score is to 1, the more the pictures are alike.

    Returns:
         how similar the generated image is to the original picture
    """
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


def evolve(population, *args):
    """
    1. Adjusts the survival rate of the population
    2. Creates children by choosing parents and combining them
    3. Mutates the polygons and sets the rate of mutation

    Returns:
        the population after all the changes
    """
    # Luck = If True, individuals randomly survive based on fitness; defaults to False.
    population.survive(fraction=0.5, luck=True)
    population.breed(parent_picker=select, combiner=combine)
    population.mutate(mutate_function=mutate, rate=0.5)  # Mutates the polygons to and sets mutation rate
    return population


if __name__ == "__main__":
    population = Population.generate(initialise, evaluate, size=10, maximize=True)
    for i in range(6000):
        evolve(population)  # returns the evolved population.
        print(i, evaluate(solution=population[0].chromosome))

    draw(population[0].chromosome).save("solution.png")
