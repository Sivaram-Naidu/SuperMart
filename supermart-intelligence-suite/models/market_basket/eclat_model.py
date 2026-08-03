import pandas as pd
from collections import defaultdict, Counter

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

TOTAL_TRANSACTIONS = len(transactions)

# =========================================================
# Build Eclat index ONCE
# =========================================================

ECLAT_INDEX = defaultdict(Counter)

for transaction in transactions:

    unique_items = set(transaction)

    for item in unique_items:
        for other in unique_items:
            if item != other:
                ECLAT_INDEX[item][other] += 1

# =========================================================
# Recommendation function
# =========================================================

def eclat_recommend(item, top_n=5):

    item = item.lower().strip()

    if item not in ECLAT_INDEX:
        return pd.DataFrame()

    total = sum(ECLAT_INDEX[item].values())

    rows = []

    for other, count in ECLAT_INDEX[item].most_common(top_n):

        rows.append([
            other,
            round(count / TOTAL_TRANSACTIONS, 4),
            round(count / total, 4)
        ])

    return pd.DataFrame(
        rows,
        columns=[
            'Recommended Product',
            'Support',
            'Confidence'
        ]
    )