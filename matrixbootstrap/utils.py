from itertools import (
    combinations_with_replacement,
)

# Function to generate all possible products and remove duplicates
def unique_products(elements) -> list:
    products = set()

    # Generate all possible pairs of elements
    for r in range(2, len(elements) + 1):
        for combo in combinations_with_replacement(elements, r):
            print(combo)
            product = 1
            for element in combo:
                product *= element
            products.add(product)

    # Optionally, convert the set back to a list
    return list(products)
