import pandas as pd
import pickle
from collections import defaultdict, Counter

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

total_transactions = len(transactions)

# Build Eclat index
eclat_index = defaultdict(Counter)

for transaction in transactions:

    unique_items = set(transaction)

    for item in unique_items:
        for other in unique_items:
            if item != other:
                eclat_index[item][other] += 1

# Save pickle file
with open('models/market_basket/eclat_index.pkl', 'wb') as f:
    pickle.dump({
        'index': dict(eclat_index),
        'total_transactions': total_transactions
    }, f)

print(' Created: models/market_basket/eclat_index.pkl')