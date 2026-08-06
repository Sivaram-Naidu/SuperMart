import pickle
import os


PICKLE_PATH = "models/ad_optimization/ad_optimization.pkl"


def load_results():

    if not os.path.exists(PICKLE_PATH):

        raise FileNotFoundError(

            "Run training/train_ad_optimization.py first."

        )

    with open(PICKLE_PATH, "rb") as file:

        results = pickle.load(file)

    return results


def get_ucb_results():

    return load_results()["ucb"]


def get_thompson_results():

    return load_results()["thompson_sampling"]


def get_best_algorithm():

    return load_results()["best_algorithm"]


if __name__ == "__main__":

    results = load_results()

    print("\n========== AD OPTIMIZATION ==========\n")

    print("Best Algorithm:")

    print(results["best_algorithm"])

    print()

    print("========== UCB ==========")

    print(

        "Total Reward:",

        results["ucb"]["total_reward"]

    )

    print(

        "Best Advertisement:",

        results["ucb"]["best_ad"]

    )

    print()

    print("===== THOMPSON SAMPLING =====")

    print(

        "Total Reward:",

        results["thompson_sampling"]["total_reward"]

    )

    print(

        "Best Advertisement:",

        results["thompson_sampling"]["best_ad"]

    )