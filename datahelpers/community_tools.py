from .constants import up, dn, ye
import datetime
from numpy import array
from pandas import DataFrame
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


def split_communities_cluster(edges, weights, cutoff_frac=0.1, unique_edges=True, verbose=False):

    if unique_edges:
        edges = list(set([x if x[0] <= x[1] else (x[1], x[0]) for x in edges]))
    # construct graph on edges
    g = ig.Graph(edges, directed=True)

    # sort connected components by size, decreasing order
    cc = sorted(g.clusters(), key=lambda x: len(x), reverse=True)

    # large connected components will be split into communities
    cc_to_communize = list(filter(lambda x: len(x) > cutoff_frac * len(g.vs), cc))

    # small connected components will be aggregated
    cc_to_aggregate = list(filter(lambda x: len(x) <= cutoff_frac * len(g.vs), cc))

    if verbose:
        print('number of nodes {0}, number of edges {1}'.format(len(g.vs), len(g.es)))

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

        # {edge of g : weight} dictionary
        w_dict = dict(zip(edges, weights))

        # edge order of g0
        ee_g0 = [e.tuple for e in g0.es]

        # weights of g0 in the order of g0
        w_g0 = [w_dict[(g0_node_g_node_conv[a], g0_node_g_node_conv[b])] for a, b in ee_g0]

        if verbose:
            print(g0.summary())
        dt = datetime.datetime.now()
        communities = g0.community_infomap(edge_weights=w_g0)
        dt2 = datetime.datetime.now()
        cur_seconds = (dt2 - dt).total_seconds()
        total_seconds += cur_seconds
        mbsp = communities.membership
        mbsp_agg.extend([(g0_node_g_node_conv[k], mbsp[k] + community_count)
                         for k in g0_node_g_node_conv.keys()])
        community_count += len(communities)
    if verbose:
        print('{0:.2f} sec on comm detection'.format(total_seconds))

    if verbose:
        print('size of communal component: {0}'.format(len(mbsp_agg)))

    mbsp_agg.extend([(n, community_count) for n in diffuse_component])

    if verbose:
        return mbsp_agg, total_seconds
    else:
        return mbsp_agg

