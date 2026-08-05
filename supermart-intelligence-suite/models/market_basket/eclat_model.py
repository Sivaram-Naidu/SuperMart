import pandas as pd
import pickle
from collections import Counter

# =========================================================
# LOAD PRETRAINED ECLAT INDEX
# =========================================================

with open('models/market_basket/eclat_index.pkl', 'rb') as f:
    data = pickle.load(f)

ECLAT_INDEX = data['index']
TOTAL_TRANSACTIONS = data['total_transactions']

# =========================================================
# ECLAT RECOMMENDATION FUNCTION
# =========================================================

def eclat_recommend(item, top_n=5):

    item = item.lower().strip()

    if item not in ECLAT_INDEX:
        return pd.DataFrame()

    total = sum(ECLAT_INDEX[item].values())

    rows = []

    for other, count in Counter(ECLAT_INDEX[item]).most_common(top_n):

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