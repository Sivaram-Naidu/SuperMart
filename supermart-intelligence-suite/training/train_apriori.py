import pandas as pd
from apyori import apriori
from collections import defaultdict

from models.market_basket.apriori_index import APRIORI_INDEX

DATASET_PATH = 'data/Market_Basket_Optimisation.csv'

# =========================================================
# Build Apriori index
# =========================================================

def train_apriori():

    dataset = pd.read_csv(DATASET_PATH, header=None)

    transactions = []

    for i in range(len(dataset)):

        transaction = [
            str(item).strip().lower()
            for item in dataset.iloc[i].dropna().tolist()
        ]

        if transaction:
            transactions.append(transaction)

    rules = apriori(
        transactions=transactions,
        min_support=0.003,
        min_confidence=0.2,
        min_lift=1.5,
        min_length=2,
        max_length=2
    )

    APRIORI_INDEX.clear()

    temp_index = defaultdict(list)

    for result in list(rules):

        support = result.support

        for rule in result.ordered_statistics:

            lhs = tuple(rule.items_base)
            rhs = tuple(rule.items_add)

            if len(lhs) == 1 and len(rhs) == 1:

                temp_index[lhs[0]].append({
                    'product': rhs[0],
                    'support': support,
                    'confidence': rule.confidence,
                    'lift': rule.lift
                })

    for item in temp_index:

        APRIORI_INDEX[item] = sorted(
            temp_index[item],
            key=lambda x: x['lift'],
            reverse=True
        )

    print('Apriori index built successfully')

# =========================================================
# Run directly
# =========================================================

if __name__ == '__main__':
    train_apriori()