from models.market_basket.apriori_model import apriori_recommend
from models.market_basket.eclat_model import eclat_recommend

# =========================================================
# Main recommendation API
# =========================================================

def get_recommendations(item, algorithm='apriori', top_n=5):

    if algorithm == 'apriori':
        return apriori_recommend(item, top_n)

    elif algorithm == 'eclat':
        return eclat_recommend(item, top_n)

    else:
        raise ValueError("Algorithm must be 'apriori' or 'eclat'")


# =========================================================
# Console test
# =========================================================

if __name__ == '__main__':

    while True:

        print("\nChoose algorithm:")
        print("1. Apriori")
        print("2. Eclat")
        print("3. Exit")

        choice = input("Enter choice: ").strip()

        if choice == '3':
            break

        product = input("Enter a product: ").strip()

        algorithm = 'apriori' if choice == '1' else 'eclat'

        result = get_recommendations(product, algorithm)

        if result.empty:
            print(f"\nNo recommendations found for '{product}' using {algorithm}.")
        else:
            print(f"\nRecommendations for '{product}' using {algorithm}:\n")
            print(result.to_string(index=False))