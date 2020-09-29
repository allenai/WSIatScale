from efficient_apriori import itemsets_from_transactions

def run_apriori(reps_to_instances, min_support):
    keys = [k for k, v in reps_to_instances.data.items() for _ in range(len(v))]
    itemsets, _ = itemsets_from_transactions(keys, min_support=min_support, output_transaction_ids=True)
    return itemsets
