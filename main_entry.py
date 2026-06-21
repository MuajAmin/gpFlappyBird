"""
Entrance of the program.
"""
from game import *
from postprocessing import *
import random
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PP_DIR = os.path.join(_BASE_DIR, 'pp')


def main():
    random.seed(RANDOM_SEED)
    game = Game()
    while game.running and game.current_generation < N_GEN:
        game.reset()
        game.run()

    if PP_FORMULA or PP_GRAPH_VISUALIZATION:
        os.makedirs(_PP_DIR, exist_ok=True)
        gs = [extract_computational_subgraph(ind) for ind in game.pop]
        # note that only the MU parents have been evaluated and have fitness values
        if PP_FORMULA:
            formula_path = os.path.join(_PP_DIR, 'formula.txt')
            print(f"Writing formula to {formula_path} ...")
            with open(formula_path, 'w') as f:
                for i, g in enumerate(gs):
                    formula = simplify(g, ['v', 'h', 'g'])
                    formula = round_expr(formula, PP_FORMULA_NUM_DIGITS)
                    print(
                        f"{i}\n score: {game.pop[i].fitness}\n formula: {formula}")
                    f.write(
                        f"{i}\n score: {game.pop[i].fitness}\n formula: {formula}\n")
        if PP_GRAPH_VISUALIZATION:
            print(f"Drawing graphs to files in folder {_PP_DIR} ...")
            for i, g in enumerate(gs):
                visualize(g, os.path.join(_PP_DIR, f"g{i}.pdf"), input_names=['v', 'h', 'g'])


if __name__ == '__main__':
    main()
