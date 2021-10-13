from .partition import partition


def apply_map_ab(map_ab, sublist):
    """
    given a multivalued map a -> b and a list of a's
    provide a list b's
    (extracting multivalues, so they don't come up again)
    :param map_ab:
    :param sublist:
    :return:
    """
    sub = []
    for a in sublist:
        if map_ab[a]:
            b = map_ab[a].pop()
            sub.append(b)
        else:
            raise ValueError(
                "a to b multivalued function "
                "and partition list of a's are misaligned"
            )
    return sub


def process_map_ab_superlist(map_ab, superlist):
    data_acc = []
    for sublist in superlist:
        sub = apply_map_ab(map_ab, sublist)
        data_acc.append(sub)
    return data_acc


def split_to_subsamples(metric_dict, irows=[], size=1000):
    if irows:
        # data_dict -> data_dicts
        list_data_dicts = []
    else:
        ids = list(metric_dict.keys())
        weights = [metric_dict[i] for i in ids]

        ids_weights = dict(zip(ids, weights))
        #         print('ids: ', sorted(ids)[:10])
        #         print('weights: ', sorted(weights)[:10])
        weights_ids = {w: [] for w in set(weights)}
        for k, w in ids_weights.items():
            weights_ids[w].append(k)
        # print(list(weights_ids.items())[:10])
        parts = partition(weights, size + 1)
        # print(list(map(sum, parts)), list(map(len, parts)))

        ids_part = process_map_ab_superlist(weights_ids, parts)
        # flat = [x for s in ids_part for x in s]
        #         print(sorted(flat)[:10])

        #         print(list(map(len, ids_part)), ids_part[0][:5])
        # list_data_dicts = [{k: metric_dict[k] for k in sub} for sub in ids_part]
    return ids_part
