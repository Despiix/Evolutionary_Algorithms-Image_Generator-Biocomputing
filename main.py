"""
Runs the evolution for a given target image while logging the
population statistics.


Usage: run.py [options] <target-file>
       run.py -h | --help

Options:
  -s NUM, --seed=NUM    seed for a random number generator [default: 31]
  -j NUM, --jobs=NUM    number of parallel jobs [default: 1]

Logging options:
  -t NUM, --step=NUM    log every NUM generation [default: 10]
  -l NAME, --log=NAME   name of the log file

Evolution parameters:
  --pop-size=NUM        number of solutions in population [default: 100]
  --generations=NUM     number of evolution iterations [default: 500]

"""

import sys
import random
import atexit
import multiprocessing

import PIL
import evol

from evol import Population
from PIL import Image, ImageChops, ImageDraw
from pathlib import Path
from docopt import docopt

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


# For parallel evaluation on win/macOS, target image has to be read
# before declaration of the evaluate function. This is due to no option
# to fork the process. Instead, a new process is created and code is
# imported. This results in ignoring any later assignments to TARGET.
# We could pass the image around, instead of using a module but that
# significantly slows down the evaluation process. Therefore, unless
# we switch to GNU/Linux, we have to suffer this ugliness here.
args = docopt(__doc__)
path = Path(args["<target-file>"])
# check if file exists
if not path.exists():
    print("Cannot find", path, file=sys.stderr)
    exit(1)
# read the target image
TARGET = PIL.Image.open(path)
TARGET.load()  # closes the file, needed for parallel eval


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


def evolve(population, generation, max_generations, *args):
    """
    1. Adjusts the survival rate of the population
    2. Creates children by choosing parents and combining them
    3. Mutates the polygons and sets the rate of mutation, which decreases as
    the number of generations increases.

    Returns:
        Population: The evolved population
    """
    start_rate = 0.9  # The initial mutation rate
    end_rate = 0.01  # The final mutation rate
    # Calculate the mutation rate for the current generation
    rate = start_rate * (1 - (generation / max_generations)) + end_rate * (generation / max_generations)

    population.survive(fraction=0.1)
    population.breed(parent_picker=select, combiner=combine)
    population.mutate(mutate_function=mutate, rate=rate)  # Use the dynamic mutation rate
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


class Population(evol.population.BasePopulation):
    def __init__(self, initialise, size, parallel_map=None):
        chromosomes = [initialise() for i in range(size)]
        super().__init__(chromosomes, None, maximize=True)
        self.apply = parallel_map or (lambda f, x: list(map(f, x)))
        self.evals = 0

    def evaluate(self, lazy=True):
        offspring = [x for x in self.individuals if x.fitness is None]

        if offspring:
            scores = self.apply(evaluate, (x.chromosome for x in offspring))
            for ind, score in zip(offspring, scores):
                ind.fitness = score
            self.evals += len(scores)

        return self


class Logger(evol.logger.BaseLogger):
    def __init__(self, target=None, stdout=False, step=1):
        if target:
            target = Path(target).absolute()
            target.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(target, stdout)

        self.step = step
        self.count = 0

    def log(self, population, *, generation):
        self.count += 1
        if self.count >= self.step:
            self.count = 0
            values = [i.fitness for i in population]
            stats = [generation, population.evals, round(min(values), 3), round(max(values), 3)]
            self.logger.info(",".join(map(str, stats)))


if __name__ == "__main__":
    # setup logging
    log_step = int(args["--step"])
    if log_file := args["--log"]:
        logger = Logger(step=log_step, target=log_file)
    else:
        logger = Logger(step=log_step, stdout=True)

    # fix the RNG seed for reproducibility
    random.seed(int(args["--seed"]))

    # setup parallel evaluation
    jobs = int(args["--jobs"])
    parallel_map = None
    if jobs > 1:
        pool = multiprocessing.Pool(jobs)
        parallel_map = pool.map
        atexit.register(pool.close)

    # create the first population
    population = Population(initialise, int(args["--pop-size"]), parallel_map)

    # Define threshold and limit for stability
    stability_threshold = 0.00001  # Adjust as needed for what you consider a "stable" change
    stability_limit = 100  # Number of generations with minimal change before stopping

    # Initialize variables to track stability
    previous_best_score = None
    stable_generations_count = 0

    for i in range(1, int(args["--generations"])):
        evolve(population, i, int(args["--generations"]), args).callback(logger.log, generation=i)
        current_best_score = population.current_best.fitness

        # Check if the score has changed significantly
        if previous_best_score is not None:
            score_change = abs(current_best_score - previous_best_score)
            if score_change < stability_threshold:
                stable_generations_count += 1
            else:
                stable_generations_count = 0  # Reset count if there's significant change

            # Check if stability limit is reached
            if stable_generations_count >= stability_limit:
                print(f"Evolution stopped due to stability at generation {i}.")
                break  # Stop the loop
        previous_best_score = current_best_score

    # Save the best solution as image outside the loop
    name = f"best_{path.stem}_p{args['--pop-size']}_g{args['--generations']}"
    draw(population.current_best.chromosome).save(f"{name}.png")
    
