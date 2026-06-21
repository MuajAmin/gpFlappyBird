"""
Main entry point for the Flappy Bird AI using NEAT
(NeuroEvolution of Augmenting Topologies).

How to run:
    python main_neat.py

Controls:
    Ctrl+1/2/3 : Change game speed (1x/2x/3x)
    Ctrl+P     : Pause
    Ctrl+H     : Add a human player bird (blue)
    Ctrl+M     : Toggle music
    Space/Up   : Flap (human player only)
"""

import os
import neat
import random
from game import *

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, 'config-feedforward')
_CHECKPOINT_DIR = os.path.join(_BASE_DIR, 'neat_checkpoints')

# Module-level game instance shared between eval_genomes() and main()
game = None


def eval_genomes(genomes, config):
    """
    Fitness evaluation function called each NEAT generation.
    Resets the game, runs the NEAT birds, and ends when all birds die.
    """
    global game
    game.reset_neat(genomes, config)
    game.run()

    if not game.running:
        # user closed the window; raise to stop neat evolution
        raise SystemExit


def run_neat(config_path):
    """
    Load NEAT config, create a Population and run evolution.
    """
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create the population
    population = neat.Population(config)

    # Add reporters so we can see progress in the console
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Optional: save a checkpoint every 5 generations
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=5,
            time_interval_seconds=None,
            filename_prefix=os.path.join(_CHECKPOINT_DIR, 'neat-checkpoint-'),
        )
    )

    # Run until a winner is found or the user closes the window
    try:
        winner = population.run(eval_genomes, n=200)
        print('\nBest genome:\n{!s}'.format(winner))
    except SystemExit:
        print('\nGame window closed. Stopping NEAT evolution.')


def main():
    global game
    random.seed(None)  # use system time for seed (unique each run)
    game = Game()
    run_neat(_CONFIG_PATH)


if __name__ == '__main__':
    main()
