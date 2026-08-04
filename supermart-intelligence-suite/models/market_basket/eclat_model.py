import pandas as pd
import models.market_basket.eclat_index as eclat_data

# =========================================================
# Recommendation function
# =========================================================

def eclat_recommend(item, top_n=5):

    item = item.lower().strip()

    if item not in eclat_data.ECLAT_INDEX:
        return pd.DataFrame()

    total = sum(eclat_data.ECLAT_INDEX[item].values())

    rows = []

    for other, count in eclat_data.ECLAT_INDEX[item].most_common(top_n):

        rows.append([
            other,
            round(count / eclat_data.TOTAL_TRANSACTIONS, 4),
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