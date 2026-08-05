import pandas as pd
import pickle
from apyori import apriori
from collections import defaultdict

# Load dataset
dataset = pd.read_csv('data/Market_Basket_Optimisation.csv', header=None)

transactions = []

for i in range(len(dataset)):
    transaction = [
        str(item).strip().lower()
        for item in dataset.iloc[i].dropna().tolist()
    ]

    if transaction:
        transactions.append(transaction)

# Run Apriori
rules = apriori(
    transactions=transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=1.5,
    min_length=2,
    max_length=2
)

apriori_index = defaultdict(list)

for result in list(rules):

    support = result.support

    for rule in result.ordered_statistics:

        lhs = tuple(rule.items_base)
        rhs = tuple(rule.items_add)

        if len(lhs) == 1 and len(rhs) == 1:

            apriori_index[lhs[0]].append({
                'product': rhs[0],
                'support': support,
                'confidence': rule.confidence,
                'lift': rule.lift
            })

# Sort by lift
for item in apriori_index:
    apriori_index[item] = sorted(
        apriori_index[item],
        key=lambda x: x['lift'],
        reverse=True
    )

# Save pickle file
with open('models/market_basket/apriori_index.pkl', 'wb') as f:
    pickle.dump(dict(apriori_index), f)

print(' Created: models/market_basket/apriori_index.pkl')