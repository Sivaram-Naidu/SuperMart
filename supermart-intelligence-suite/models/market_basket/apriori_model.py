import pandas as pd
from apyori import apriori
from collections import defaultdict

DATASET_PATH = 'data/Market_Basket_Optimisation.csv'

# =========================================================
# Load transactions
# =========================================================

dataset = pd.read_csv(DATASET_PATH, header=None)

transactions = []

for i in range(len(dataset)):

    transaction = [
        str(item).strip().lower()
        for item in dataset.iloc[i].dropna().tolist()
    ]

    if transaction:
        transactions.append(transaction)

# =========================================================
# Build Apriori index ONCE
# =========================================================

rules = apriori(
    transactions=transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=1.5,
    min_length=2,
    max_length=2
)

APRIORI_INDEX = defaultdict(list)

for result in list(rules):

    support = result.support

    for rule in result.ordered_statistics:

        lhs = tuple(rule.items_base)
        rhs = tuple(rule.items_add)

        if len(lhs) == 1 and len(rhs) == 1:

            APRIORI_INDEX[lhs[0]].append({
                'product': rhs[0],
                'support': support,
                'confidence': rule.confidence,
                'lift': rule.lift
            })

# Sort by lift
for item in APRIORI_INDEX:
    APRIORI_INDEX[item] = sorted(
        APRIORI_INDEX[item],
        key=lambda x: x['lift'],
        reverse=True
    )

# =========================================================
# Recommendation function
# =========================================================

def apriori_recommend(item, top_n=5):

    item = item.lower().strip()

    if item not in APRIORI_INDEX:
        return pd.DataFrame()

    rows = []

    for r in APRIORI_INDEX[item][:top_n]:

        rows.append([
            r['product'],
            round(r['support'], 4),
            round(r['confidence'], 4),
            round(r['lift'], 4)
        ])

    return pd.DataFrame(
        rows,
        columns=[
            'Recommended Product',
            'Support',
            'Confidence',
            'Lift'
        ]
    )