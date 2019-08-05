def group_by(items, key_func):
    "itertools.groupby is scary because it can only be iterated over once and I don't need to be super efficient for just preprocessing"
    grouped_items = dict()
    for obj in items:
        key = key_func(obj)
        if key not in grouped_items:
            grouped_items[key] = []
        grouped_items[key].append(obj)
    return grouped_items