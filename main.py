import random
import numpy as np

from evol import Population
from PIL import Image, ImageDraw

SIDES = 3
SIDES2 = 7
POLYGON_COUNT = 100
# Constants for image
MAX = 255 * 200 * 200


# TARGET = Image.open("Images/3a.png")
# TARGET.load()  # read image and close the file


# Make the polygon and randomise the coordinates of the polygon corners
def make_polygon(n):
    """
    Randomly assigns the colour, opaqueness and coordinates of the polygons in this instance
     I chose to use triangles because I believe they will be an easier shape to work with
    """
    R, G, B = [random.randrange(256) for i in range(3)]  # Assigns random int values to Red, Green and Blue
    A = random.randrange(30, 60)  # Sets random opaqueness
    x_y_coords = [random.randrange(10, 190) for i in range(n * 2)]  # Sets random coordinates to polygons

    # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190
    return [(R, G, B, A)] + [(x_y_coords[i], x_y_coords[i + 1]) for i in range(0, len(x_y_coords), 2)]


# experimenting with mixed shape types
def initialise():
    """
    Assigns the sides of the polygon and creates as many polygons as stated in the constants above,
    with 70% of the polygons having 'SIDES' sides and 30% having 'SIDES2' sides.
    """
    polygon_count_sides = int(POLYGON_COUNT * 0.8)  # Percentage of polygons with 'SIDES' sides
    polygon_count_sides2 = POLYGON_COUNT - polygon_count_sides  # Ensures the total counts up to POLYGON_COUNT

    polygons = [make_polygon(SIDES) for _ in range(polygon_count_sides)]
    polygons += [make_polygon(SIDES2) for _ in range(polygon_count_sides2)]
    return polygons


# Initialise function when needing to use 1 shape type instead of two
'''
def initialise():
    """Assigns the sides of the polygon and creates as many polygons as stated in the constants above"""
    return [make_polygon(SIDES) for i in range(POLYGON_COUNT)]
'''


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


def tournament_select(population, tournament_size=2):
    """
    Selects an individual via tournament selection.
    Parameters:
    - population: The current population from which to select.
    - tournament_size: The number of individuals to compete in each tournament.

    Returns:
    - The winner of the tournament.
    """
    # Randomly select `tournament_size` individuals for the tournament
    tournament_participants = random.sample(population, tournament_size)

    # Select the individual with the best fitness
    winner = max(tournament_participants, key=lambda individual: individual.fitness)
    return winner


def select(population):
    """Selects 2 parents using tournament selection."""
    return [tournament_select(population) for _ in range(2)]


def combine(*parents):
    """Combines the parents chosen in select to create a child"""
    return [a if random.random() < 0.5 else b for a, b in zip(*parents)]


# Sine wave mutation to allow for both, exploration and exploitation
def sine_func(gen, gens_per_cycle=200, decay=0.00001, min_=0.1):
    return np.maximum(np.sin(gen * (2 * np.pi) / gens_per_cycle) ** 2 * np.exp(-gen * decay), min_)


# generation, max_generations are there for the use of dynamic mutation rates
def evolve(population, generation, max_generations, *args):
    """
    1. Adjusts the survival rate of the population
    2. Creates children by choosing parents and combining them
    3. Mutates the polygons and sets the rate of mutation, which decreases as
    the number of generations increases.

    Returns:
        Population: The evolved population
    """
    # Start, end rates and rate give the option to change to simulated annealing approach if needed

    start_rate = 0.9  # The initial mutation rate
    end_rate = 0.01  # The final mutation rate
    # Calculate the mutation rate for the current generation
    rate = start_rate * (1 - (generation / max_generations)) + end_rate * (generation / max_generations)

    population.survive(fraction=0.4)
    population.breed(parent_picker=select, combiner=combine)
    # uncomment code above and type rate=rate to use simulated annealing
    population.mutate(mutate_function=mutate, rate=rate)
    return population


'''
# My initial code for the evolution algorithm - does not use Ipython
if __name__ == "__main__":
    population = Population.generate(initialise, evaluate, size=10, maximize=True)
    for i in range(5000):
        evolve(population)  # returns the evolved population.
        print(i, evaluate(solution=population[0].chromosome))

    draw(population[0].chromosome).save("solution.png")
'''