from .constants import up, dn, ye
import datetime
from itertools import product
from numpy import array, percentile
from pandas import DataFrame, read_hdf, concat, HDFStore
from os.path import expanduser, join
import igraph as ig


def project_weight(dft):
    edges_year = dft[[up, dn, ye]].groupby([up, dn, ye]).apply(lambda x: x.shape[0])
    edges = edges_year.groupby(level=(0, 1)).apply(lambda x: x.sum())
    return edges.reset_index().rename(columns={0: 'weight'})


def cut_edges(edges, nnodes=False, nedges=False, verbose=False):
    edges2 = edges.copy()

    if verbose:
        print(edges2.shape)
    ups, dns = edges2[up].unique(), edges2[dn].unique()
    prots = list(set(ups) | set(dns))

    if verbose:
        print('{0} unique nodes'.format(len(prots)))
    if nnodes:
        cut_prots = prots[:nnodes]
    else:
        cut_prots = prots
    if verbose:
        print('{0} cut unique nodes'.format(len(cut_prots)))
    if nnodes:
        mask = (edges2[up].isin(cut_prots)) & (edges2[dn].isin(cut_prots))
        edges2 = edges2.loc[mask]
        if verbose:
            print('so with nodes cut : we {0} edges'.format(edges2.shape[0]))
    if nedges:
        edges2 = edges2.head(nedges)
    ups, dns = edges2[up].unique(), edges2[dn].unique()
    if verbose:
        print(len(set(ups)), len(set(dns)))
    prots = list(set(ups) | set(dns))
    if verbose:
        print(len(prots), edges2.shape, edges.shape)
    int_to_prot = {k: prots[k] for k in range(len(prots))}
    prot_to_int = {int_to_prot[k]: k for k in int_to_prot.keys()}

    g_edges = list(map(lambda x: (prot_to_int[x[0]], prot_to_int[x[1]]), edges2[[up, dn]].values))
    # weights has the same order as edges
    g_edges_weights = list(edges2['weight'])
    g_vertices = prot_to_int.values()
    return g_edges, g_vertices, int_to_prot, prot_to_int, g_edges_weights


def optimal_nclusters(communities, frac=0.1, verbose=False):
    k = 0
    flag = True
    while flag:
        k += 1
        clusters = communities.as_clustering(k)
        lens = [len(c) for c in clusters]
        if min(lens) < frac*max(lens):
            if verbose:
                print('suboptimal community structure:', lens)
            flag = False
    return k-1


def produce_cluster_df(edges_df, cutoff_frac=0.1, unique_edges=True, cut_nodes=False, verbose=False):

    # edges_df has up, dn and weight columns
    if verbose:
        print('number of edges: {0}'.format(edges_df.shape[0]))

    # choose subset graph with predefined number of nodes and edges
    ee, vv, i2p, p2i, weights = cut_edges(edges_df, nnodes=cut_nodes)
    if verbose:
        print('number of edges after cutting: {0}'.format(len(ee)))

    r = split_communities_cluster(ee, weights, cutoff_frac, unique_edges, verbose)

    if verbose:
        mbsp_agg, seconds = r
    else:
        mbsp_agg = r

    domain_coding_df = DataFrame(array(mbsp_agg), columns=['int_id', 'domain_id'])
    if verbose:
        print('shape of domain_coding_df : {0}'.format(domain_coding_df.shape))

    domain_coding_df['id'] = domain_coding_df['int_id'].apply(lambda x: i2p[x])

    dff = edges_df.merge(domain_coding_df[['id', 'domain_id']], how='inner', left_on=up, right_on='id')
    del dff['id']
    dff2 = dff.merge(domain_coding_df[['id', 'domain_id']], how='inner',
                     left_on=dn, right_on='id', suffixes=('_up', '_dn'))
    del dff2['id']
    if verbose:
        return dff2, seconds
    else:
        return dff2


def split_communities_cluster(edges, weights=None, cutoff_frac=0.1, directed=True, verbose=False):
    """

    :param edges: list of tuples [(a,b)]
    :param weights: weights pof edges, the order is assumed to be the same that of edges
    :param cutoff_frac:
    :param directed:
    :param verbose:
    :return:
    """

    if not directed:
        edges = list(set([x if x[0] <= x[1] else (x[1], x[0]) for x in edges]))
    # construct graph on edges
    g = ig.Graph(edges, directed=directed)

    # sort connected components by size, decreasing order
    cc = sorted(g.clusters(), key=lambda x: len(x), reverse=True)
    print('number of connected components {0}'.format(len(cc)))

    # large connected components will be split into communities
    cc_to_communize = list(filter(lambda x: len(x) > cutoff_frac * len(g.vs), cc))

    # small connected components will be aggregated
    cc_to_aggregate = list(filter(lambda x: len(x) <= cutoff_frac * len(g.vs), cc))

    if verbose:
        print('number of nodes {0}, number of edges {1}'.format(len(g.vs), len(g.es)))

        print('number of connected components {0}'.format(len(cc)))
        print('number of disconnected components with > {0} '
              'fraction of nodes : {1}'.format(cutoff_frac, len(cc_to_communize)))
        print('number of disconnected components with <= {0} '
              'fraction of nodes : {1}'.format(cutoff_frac, len(cc_to_aggregate)))
        sum_small_disconnected = sum(map(lambda x: len(x), cc_to_aggregate))
        print('sum of small disconnected components: {0}, '
              'frac of total number of nodes {1:.3f}'.format(sum_small_disconnected,
                                                             float(sum_small_disconnected) / len(g.vs)))

    diffuse_component = [node for cc0 in cc_to_aggregate for node in cc0]
    if verbose:
        print('size of diffuse component: {0}'.format(len(diffuse_component)))

    community_count = 0
    mbsp_agg = []
    total_seconds = 0
    for c in cc_to_communize:
        # TODO redo the hacky part
        # hack to keep track of labels
        if verbose:
            print('community count : {0}'.format(community_count))

        g0_node_g_node_conv = dict(zip(range(len(c)), c))

        # subgraph() disrespects the ids of c and creates ids from 0 conseq.
        g0 = g.subgraph(c)
        if verbose:
            print(g0.summary())
        dt = datetime.datetime.now()

        # edge order of g0
        ee_g0 = [e.tuple for e in g0.es]

        if weights:
            # {edge of g : weight} dictionary
            w_dict = dict(zip(edges, weights))
            # weights of g0 in the order of g0
            w_g0 = [w_dict[(g0_node_g_node_conv[a], g0_node_g_node_conv[b])] for a, b in ee_g0]
            communities = g0.community_infomap(edge_weights=w_g0)
        else:
            communities = g0.community_infomap()

        dt2 = datetime.datetime.now()
        cur_seconds = (dt2 - dt).total_seconds()
        total_seconds += cur_seconds
        mbsp = communities.membership
        mbsp_agg.extend([(g0_node_g_node_conv[k], mbsp[k] + community_count)
                         for k in g0_node_g_node_conv.keys()])
        community_count += len(communities)

    if verbose:
        print('{0:.2f} sec on comm detection'.format(total_seconds))
        print('size of communal component: {0}'.format(len(mbsp_agg)))

    mbsp_agg.extend([(n, community_count) for n in diffuse_component])

    if verbose:
        return mbsp_agg, total_seconds
    else:
        return mbsp_agg


def assign_comms_to_edge_list(elist, directed=True):
    ups = [x[0] for x in elist]
    dns = [x[1] for x in elist]
    uni_nodes = list(set(ups) | set(dns))
    n_uniques = len(uni_nodes)
    ranged = range(n_uniques)
    conversion_map = dict(zip(uni_nodes, ranged))
    edges_list_conv = [(conversion_map[x[0]], conversion_map[x[1]]) for x in elist]
    g = ig.Graph(edges=edges_list_conv, directed=directed)
    communities = g.community_infomap()


def prepare_graph_from_df(df, file_format='matrix',
                          directed=False, percentile_value=None, verbose=False):

    """
    given a DataFrame in either edges or adj matrix format spit out a igraph
    :param df:
    :param file_format:
    :param directed:
    :param percentile_value:
    :param multi_flag:
    :param verbose:
    :return:
    """

    if file_format == 'matrix':
            ups = set(df.index)
            dns = set(df.columns)
            if verbose:
                print('max of ups: {0}; max of dns: {1}'.format(max(list(ups)), max(list(dns))))
            uni_nodes = list(set(ups) | set(dns))
            n_uniques = len(uni_nodes)
            conversion_map = dict(zip(uni_nodes, range(n_uniques)))
            inv_conversion_map = dict(zip(range(n_uniques), uni_nodes))
            df2 = df.rename(columns=conversion_map, index=conversion_map)
            if verbose:
                print('max of renamed columns: {0}; max of renamed index: {1}'.format(max(df2.columns),
                                                                                      max(df2.index)))
            df2 = df2.stack()
            df2 = df2.abs()
            df2 = df2.replace({0: 1e-6})
            if verbose:
                print(df2.head())
    elif file_format == 'edges':
        c1, c2, c3 = df.columns[:3]
        df = df.groupby([c1, c2]).apply(lambda x: x.loc[:, c3].max()).reset_index()
        ups = set(df[c1])
        dns = set(df[c2])
        uni_nodes = list(set(ups) | set(dns))
        n_uniques = len(uni_nodes)
        conversion_map = dict(zip(uni_nodes, range(n_uniques)))
        inv_conversion_map = dict(zip(range(n_uniques), uni_nodes))
        df2 = df.copy()
        df2[c1] = df2[c1].apply(lambda x: conversion_map[x])
        df2[c2] = df2[c2].apply(lambda x: conversion_map[x])
        df2 = df2.set_index([c1, c2])
    else:
        return None

    if percentile_value:
        thr = percentile(df2, percentile_value)
        df2 = df2[df2 > thr]

    edges, weights = df2.index.tolist(), df2.values
    g = ig.Graph(edges, directed=directed)

    return g, weights, inv_conversion_map


def prepare_graphdf(full_fname, file_format='matrix', key_hdf_store=False, verbose=False):
    if file_format == 'matrix':
        df = read_hdf(expanduser(full_fname))
    elif file_format == 'edges':
        if isinstance(full_fname, str) and not isinstance(full_fname, list):
            if key_hdf_store:
                store = HDFStore(expanduser(full_fname), mode='r')
                keys = store.keys()
                str_key = [k for k in keys if str(key_hdf_store) in k][0]
                df = read_hdf(expanduser(full_fname), key=str_key)
            else:
                df = read_hdf(expanduser(full_fname))
        elif isinstance(full_fname, list) and not isinstance(full_fname, str):
            df = []
            for f in full_fname:
                if key_hdf_store:
                    store = HDFStore(expanduser(f), mode='r')
                    keys = store.keys()
                    str_keys = [k for k in keys if str(key_hdf_store) in k]
                    if str_keys:
                        df_ = read_hdf(expanduser(f), key=str_keys[0], mode='r')
                    else:
                        df_ = DataFrame()
                    store.close()
                    if verbose:
                        print('opened {0} by key {1}. Shape is {2}'.format(f, key_hdf_store, df_.shape))
                else:
                    df_ = read_hdf(expanduser(f))
                df.append(df_)
            df = concat(df)
        else:
            df = None
    else:
        df = None
    return df


def graph_to_comms(g, ws, inv_conversion_map, fpath_out, method='multilevel',
                   directed=False, weighted=False, percentile_value=None, origin=None,
                   key_hdf_store=False, verbose=False):

    dt = datetime.datetime.now()
    total_seconds = 0

    if method == 'multilevel':
        if directed:
            raise ValueError('multilevel can not be directed')
        if weighted:
            communities = g.community_multilevel(weights=ws)
        else:
            communities = g.community_multilevel()
    elif method == 'infomap':
        if weighted:
            communities = g.community_infomap(edge_weights=ws)
        else:
            communities = g.community_infomap()
    else:
        return None

    dt2 = datetime.datetime.now()
    cur_seconds = (dt2 - dt).total_seconds()
    total_seconds += cur_seconds
    if verbose:
        print('compute took {0} sec; number of communities: {1}'.format(total_seconds, len(communities)))
        lens = sorted([len(c) for c in communities])
        print('Largest 5 are {0}'.format(lens[-5:]))
        print('3 top elements from 3 comms are {0}'.format([x for x in communities.membership[:3]]))
        print('3 top vertices are {0}'.format([x for x in g.vs[:3]]))
    comm_df = DataFrame(communities.membership, index=[v.index for v in g.vs], columns=['comm_id'])
    if verbose:
        print('comm_df index max : {0}'.format(comm_df.index.max()))
    comm_df = comm_df.rename(index=inv_conversion_map).sort_index()
    if verbose:
        print('comm_df index max : {0}'.format(comm_df.index.max()))
        print('comm_df head : {0}'.format(comm_df.head()))
        print('comm_df tail : {0}'.format(comm_df.tail()))
    directedness = 'dir' if directed else 'undir'
    weighted = 'wei' if weighted else 'unwei'
    # if isinstance(full_fname, str) and not isinstance(full_fname, list):
    #     prefix = full_fname.split('/')[-1].split('.')[0]
    # else:
    #     prefix = full_fname[0].split('/')[-1].split('.')[0]
    if key_hdf_store:
        h5_fname = '{0}_comm_{1}_{2}_{3}_p{4}.h5'.format(origin, method,
                                                         directedness, weighted, percentile_value)
        if verbose:
            print('putting comms to {0}'.format(expanduser(join(fpath_out, h5_fname))))
        store = HDFStore(expanduser(join(fpath_out, h5_fname)))
        store.put('{0}'.format(key_hdf_store), comm_df, format='t')
        store.close()
    else:
        fout_name = '{0}_comm_{1}_{2}_{3}_p{4}.csv.gz'.format(origin, method,
                                                              directedness, weighted,
                                                              percentile_value)
        if verbose:
            print('putting comms to {0}'.format(fout_name))
        comm_df.to_csv(expanduser(join(fpath_out, fout_name)), compression='gzip')
    return cur_seconds


def calculate_comms(full_fname, fpath_out, file_format='matrix', method='multilevel',
                    directed=False, weighted=False, percentile_value=None, origin=None,
                    key_hdf_store=False, verbose=False):
    df = prepare_graphdf(full_fname, file_format, key_hdf_store, verbose)

    g, ws, inv_conversion_map = prepare_graph_from_df(df, file_format, directed, percentile_value,
                                                      verbose)
    dt = graph_to_comms(g, ws, inv_conversion_map, fpath_out, method, directed, weighted,
                        percentile_value, origin, key_hdf_store, verbose)
    return dt


#TODO rewrite using decorators
def meta_calculate_comms(full_fname, fpath_out, file_format='matrix', method='multilevel',
                         directed=False, weighted=False, percentile_value=None, origin=None,
                         run_over_keys_flag=False, verbose=False):
    if verbose:
        print('run over keys flag: {0}'.format(run_over_keys_flag))
    if run_over_keys_flag:
        # fetch all keys from potentially a list of files in full_fname
        # and loop over them
        set_keys = set()
        for f in full_fname:
            store = HDFStore(expanduser(f), mode='r')
            set_keys |= set(store.keys())
            store.close()
        keys = sorted(list(set_keys))
        tot = 0
        for k in keys[:]:
            # key corresponds to year, so truncate the prefix
            kval = 'y' + k[-4:]
            dt = calculate_comms(full_fname, fpath_out, file_format, method,
                                 directed, weighted, percentile_value, origin, kval, verbose)
            tot += dt
    else:
        tot = calculate_comms(full_fname, fpath_out, file_format, method,
                              directed, weighted, percentile_value, origin,
                              run_over_keys_flag, verbose)
    return tot


def get_community_fnames_cnames(mode='lincs', storage_type='csv.gz'):

    if storage_type != 'csv.gz':
        suffix = '_dyn'
    else:
        suffix = ''

    # arguments and column generator for lincs community detection methods
    methods = ['multilevel', 'infomap']
    directeds = [True, False]
    weighteds = [True, False]
    percentile_values = [None, 95]

    if mode == 'lincs':
        types = ['lincs']
    elif mode[:2] == 'gw':
        types = [mode]
    elif mode[:3] == 'lit':
        types = [mode]
    else:
        types = [mode]

    keys = ['method', 'directed']
    keys2 = ['weighted', 'percentile_value']
    largs = [{k: v for k, v in zip(keys, p)} for p in product(*(methods, directeds))]

    zargs = [{k: v for k, v in zip(keys2, p)} for p in zip(*(weighteds, percentile_values))]
    origins = [{'origin': '{0}'.format(t)} for t in types]

    inv_args = {'fpath_out': '~/data/kl/comms/',
                'file_format': 'matrix'}
    targs = [{**z, **l, **inv_args, **t} for l, z, t in product(largs, zargs, origins)]
    targs2 = list(filter(lambda x: not (x['directed'] and x['method'] == 'multilevel'), targs))

    if mode == 'lincs':
        #undirected infomap on full graph fails on 16Gb
        targs2 = list(filter(lambda x:
                             not (x['method'] == 'infomap' and x['directed'] is False
                                  and not x['percentile_value']), targs2))

    fnames = []
    cnames = []
    for aa in targs2:
        directedness = 'dir' if aa['directed'] else 'undir'
        weighted = 'wei' if aa['weighted'] else 'unwei'
        percentile_value = aa['percentile_value']
        mname = 'im' if aa['method'] == 'infomap' else 'ml'
        fout_name = '{0}_comm_{1}_{2}_{3}_p{4}.{5}'.format(aa['origin'], aa['method'],
                                                           directedness, weighted,
                                                           percentile_value, storage_type)
        cname_full = '{0}_comm_{1}_{2}_{3}_p{4}{5}'.format(aa['origin'], mname, directedness,
                                                           weighted, percentile_value, suffix)

        fnames.append(fout_name)
        cnames.append(cname_full)
    return fnames, cnames
