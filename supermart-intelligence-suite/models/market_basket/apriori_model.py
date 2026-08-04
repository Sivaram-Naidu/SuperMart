import pandas as pd
from models.market_basket.apriori_index import APRIORI_INDEX

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