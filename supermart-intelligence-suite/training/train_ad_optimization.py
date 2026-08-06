import os
import math
import random
import pickle
import pandas as pd


# ------------------------------
# Load Dataset
# ------------------------------
dataset = pd.read_csv("data/ads CTR optimisation.csv")

N = len(dataset)
d = len(dataset.columns)


# ==================================================
# Upper Confidence Bound
# ==================================================

ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d

total_reward = 0

for n in range(N):

    ad = 0
    max_upper_bound = 0

    for i in range(d):

        if numbers_of_selections[i] > 0:

            average_reward = (
                sums_of_rewards[i]
                / numbers_of_selections[i]
            )

            delta_i = math.sqrt(
                (3 / 2)
                * math.log(n + 1)
                / numbers_of_selections[i]
            )

            upper_bound = average_reward + delta_i

        else:

            upper_bound = float("inf")

        if upper_bound > max_upper_bound:

            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)

    numbers_of_selections[ad] += 1

    reward = dataset.iloc[n, ad]

    sums_of_rewards[ad] += reward

    total_reward += reward

ucb_result = {

    "algorithm": "UCB",

    "ads_selected": ads_selected,

    "selection_count": numbers_of_selections,

    "reward_count": sums_of_rewards,

    "total_reward": total_reward,

    "best_ad": numbers_of_selections.index(
        max(numbers_of_selections)
    ) + 1
}


# ==================================================
# Thompson Sampling
# ==================================================

ads_selected_ts = []

numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d

total_reward_ts = 0

for n in range(N):

    ad = 0
    max_random = 0

    for i in range(d):

        random_beta = random.betavariate(

            numbers_of_rewards_1[i] + 1,

            numbers_of_rewards_0[i] + 1

        )

        if random_beta > max_random:

            max_random = random_beta
            ad = i

    ads_selected_ts.append(ad)

    reward = dataset.iloc[n, ad]

    if reward == 1:

        numbers_of_rewards_1[ad] += 1

    else:

        numbers_of_rewards_0[ad] += 1

    total_reward_ts += reward

ts_result = {

    "algorithm": "Thompson Sampling",

    "ads_selected": ads_selected_ts,

    "reward_count": numbers_of_rewards_1,

    "selection_count": [

        numbers_of_rewards_1[i]
        + numbers_of_rewards_0[i]

        for i in range(d)

    ],

    "total_reward": total_reward_ts,

    "best_ad": numbers_of_rewards_1.index(
        max(numbers_of_rewards_1)
    ) + 1
}


# ==================================================
# Compare
# ==================================================

if total_reward > total_reward_ts:

    best_algorithm = "UCB"

elif total_reward_ts > total_reward:

    best_algorithm = "Thompson Sampling"

else:

    best_algorithm = "Both"


results = {

    "ucb": ucb_result,

    "thompson_sampling": ts_result,

    "best_algorithm": best_algorithm

}


# ==================================================
# Save Pickle
# ==================================================

os.makedirs("models/ad_optimization", exist_ok=True)

with open(

    "models/ad_optimization/ad_optimization.pkl",

    "wb"

) as file:

    pickle.dump(results, file)

print("✅ ad_optimization.pkl created successfully.")