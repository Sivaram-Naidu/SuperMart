import pandas as pd
import models.market_basket.eclat_index as eclat_data

DATASET_PATH = 'data/Market_Basket_Optimisation.csv'

# =========================================================
# Build Eclat index
# =========================================================

def train_eclat():

    dataset = pd.read_csv(DATASET_PATH, header=None)

    transactions = []

    for i in range(len(dataset)):

        transaction = [
            str(item).strip().lower()
            for item in dataset.iloc[i].dropna().tolist()
        ]

        if transaction:
            transactions.append(transaction)

    # Update the shared variable
    eclat_data.TOTAL_TRANSACTIONS = len(transactions)

    eclat_data.ECLAT_INDEX.clear()

    for transaction in transactions:

        unique_items = set(transaction)

        for item in unique_items:
            for other in unique_items:
                if item != other:
                    eclat_data.ECLAT_INDEX[item][other] += 1

    print(f'Eclat index built successfully with {eclat_data.TOTAL_TRANSACTIONS} transactions')

# =========================================================
# Run directly
# =========================================================

if __name__ == '__main__':
    train_eclat()