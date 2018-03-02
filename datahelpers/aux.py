from itertools import product
import argparse


def cut_string(x):
    if '_' in x:
        x2 = x.split('_')
        if '|' in x:
            # if we need hgnc ids, work with x3
            x3 = x2[0].split('|')
            r = x2[1:]
        else:
            r = x2[1:]
    else:
        r = [x]
    return r


def unfold_df(df):
    acc = []
    for i, row in df.iterrows():
        up_, dn_, act = row.values
        x, y = cut_string(up_), cut_string(dn_)
        acc.extend(list(product(*[[i], x, y, [act]])))
    return df.columns, acc


def find_closest_year(x, years):
    # what if we don't the feature (article influence, e.g.) for a give (issn, year) pair?
    # yield the closet year for the same issn!
    # add proxy_years to (pm-issn-ye) which are closest to issn-ye in (issn-ye-ai)
    # years is sorted
    left = 0
    right = len(years) - 1
    if x <= years[left]:
        return years[left]
    elif x >= years[right]:
        return years[right]
    else:
        while right - left > 1:
            mid = (right + left) // 2
            if (x - years[left]) * (years[mid] - x) > 0:
                right = mid
            else:
                left = mid
        return years[left]


def drop_duplicates_cols_arrange_col(dft, columns, col):
    # drop rows with col == 'NULL'
    # drop (ni, pm) duplicates
    # only max value of col remain from duplicates
    maskt = (dft[col] == 'NULL')
    print('fraction of claims with missing {0} '
          'dropped: {1:.4f}'.format(col, float(sum(maskt)) / maskt.shape[0]))
    df2 = dft.loc[~maskt].copy()
    df2[col] = df2[col].astype(float)
    df2 = df2.reset_index(drop=True)
    idx = df2.groupby(columns)[col].idxmax()
    df3 = df2.loc[idx]
    # df3 = df3.drop_duplicates(columns)
    print('fraction of claims (same pmid extractions) dropped: {0:.4f}'.format(1. -
                                                                               float(df3.shape[0]) / df2.shape[0]))
    return df3


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y" "1"):
        return True
    if v.lower() in ("no", "false", "f", "n" "0"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
