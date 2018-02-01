from .constants import up, dn, ye
import datetime
from numpy import concatenate, array
from pandas import DataFrame
import igraph as ig


def prepare_head(dft):
    edges_year = dft[[up, dn, ye]].groupby([up, dn, ye]).apply(lambda x: x.shape[0])
    edges = edges_year.groupby(level=(0, 1)).apply(lambda x: x.sum())
    return edges.reset_index()


def cut_edges(edges, nnodes=False, nedges=False):
    edges2 = edges.copy()
    print(edges2.shape)
    ups, dns = edges2[up].unique(), edges2[dn].unique()
    prots = list(set(ups) | set(dns))
    print('{0} unique nodes'.format(len(prots)))
    if nnodes:
        cut_prots = prots[:nnodes]
    else:
        cut_prots = prots
    print('{0} cut unique nodes'.format(len(cut_prots)))
    if nnodes:
        mask = (edges2[up].isin(cut_prots)) & (edges2[dn].isin(cut_prots))
        edges2 = edges2.loc[mask]
        print('so nodes cut : {0}'.format(edges2.shape))
    if nedges:
        edges2 = edges2.head(nedges)
    ups, dns = edges2[up].unique(), edges2[dn].unique()
    print(len(set(ups)), len(set(dns)))
    prots = list(set(ups) | set(dns))
    print(len(prots), edges2.shape, edges.shape)
    int_to_prot = {k: prots[k] for k in range(len(prots))}
    prot_to_int = {int_to_prot[k]:k for k in int_to_prot.keys()}
    g_edges = list(map(lambda x: (prot_to_int[x[0]], prot_to_int[x[1]]), edges2[[up,dn]].values))
    g_vertices = prot_to_int.values()
#     print(len(g_edges), len(g_vertices))
    return g_edges, g_vertices, int_to_prot, prot_to_int


def detect_comm(g_edges):
    g = ig.Graph(g_edges, directed=True)
    dt = datetime.datetime.now()
    communities = g.community_edge_betweenness(directed=True)
    dt2 = datetime.datetime.now()
    print((dt2-dt).total_seconds())
    clusters = communities.as_clustering(5)
    print([len(c) for c in clusters])


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


def produce_cluster_df(df, cutoff_frac=0.1, unique_edges=True, cut_nodes=False, verbose=False):
    # get edges df
    ees = prepare_head(df)

    # choose subset graph with predefined number of nodes and edges
    ee, vv, i2p, p2i = cut_edges(ees, nnodes=cut_nodes)

    if unique_edges:
        ee = list(set([x if x[0] <= x[1] else (x[1], x[0]) for x in ee]))
    # construct graph on edges
    g = ig.Graph(ee)

    # sort connected components by size, decreasing order
    cc = sorted(g.clusters(), key=lambda x: len(x), reverse=True)

    # large connected components will be split into communities

    cc_to_communize = list(filter(lambda x: len(x) > cutoff_frac * len(vv), cc))
    # small connected components will be aggregated

    cc_to_aggregate = list(filter(lambda x: len(x) <= 0.1 * len(vv), cc))

    if verbose:
        print('number of disconnected components with > {0} '
              'fraction of nodes : {1}'.format(cutoff_frac, len(cc_to_communize)))
        print('number of disconnected components with  <= {0} '
              'fraction of nodes : {1}'.format(cutoff_frac, len(cc_to_aggregate)))
        sum_small_disconnected = sum(map(lambda x: len(x), cc_to_aggregate))
        print('sum of small disconnected components des : {0}, '
              'frac of total number of nodes {1:.3f}'.format(sum_small_disconnected,
                                                             float(sum_small_disconnected) / len(vv)))

    diffuse_component = [node for cc0 in cc_to_aggregate for node in cc0]
    if verbose:
        print(len(diffuse_component))

    domains = []
    for c in cc_to_communize:
        # TODO redo the hacky part
        # hack to keep track of labels
        conv_dd = dict(zip(range(len(c)), c))

        # subgraph() disrespects the ids of c and creates ids from 0 conseq.
        g0 = g.subgraph(c)

        if verbose:
            print(g0.summary())
        dt = datetime.datetime.now()
        # communities = g0.community_edge_betweenness(directed=True)
        communities = g0.community_fastgreedy()
        dt2 = datetime.datetime.now()
        if verbose:
            print('{0:.2f} sec'.format((dt2 - dt).total_seconds()))
        k_opt = optimal_nclusters(communities, verbose=verbose)
        print(k_opt)
        clusters_ = communities.as_clustering(k_opt)
        clusters = [[conv_dd[x] for x in c_] for c_ in clusters_]
        if verbose:
            print([len(c) for c in clusters])
        ss0 = set(c)
        ss1 = set([n for sublist in clusters for n in sublist])
        print(len(ss0), len(ss1), len(ss0 & ss1))
        domains.extend(clusters)

    domains.append(diffuse_component)
    if verbose:
        print(len(domains), [len(x) for x in domains], sum([len(x) for x in domains]))
        print(len(set([el for sublist in domains for el in sublist])))

    arr = concatenate([array([(x, k) for x in domains[k]]) for k in range(len(domains))])
    print(arr.shape)
    domain_coding_df = DataFrame(arr, columns=['int_id', 'domain_id'])
    domain_coding_df['id'] = domain_coding_df['int_id'].apply(lambda x: i2p[x])
    dff = df.merge(domain_coding_df[['id', 'domain_id']], how='left', left_on=up, right_on='id')
    dff2 = dff.merge(domain_coding_df, how='left', left_on=dn, right_on='id', suffixes=('_up', '_dn'))
    return dff2