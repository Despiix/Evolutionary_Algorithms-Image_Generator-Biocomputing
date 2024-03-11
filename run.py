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

from PIL import Image, ImageChops
from pathlib import Path
from docopt import docopt

from main import draw, initialise, evolve

SIDES = 3
# Constants for image
MAX = 255 * 200 * 200
TARGET = None

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
    stability_limit = 500  # Number of generations with minimal change before stopping

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
