"""graphutils.py: A data-processing library for dealing with large graphs."""
__author__ = "Hernan Palombo"

import sys, os, time, logging, shlex
#from typing import Union, Any, Optional
import numpy  # type: ignore
import snap  # type: ignore
import random  # type: ignore
from collections import OrderedDict
from subprocess import call

metricsLT = OrderedDict(
    [('b', 'betweenness'), ('c', 'closeness'), ('d', 'degree'), ('h', 'harmonic'), ('l', 'clustering')])

TUNGraph = snap.TUNGraph  # type: Any


# calculate correlation
def correlate(d, out_file_path=''):
    """
    :param d: a dictionary <metric, <nid, val>>
    :type d: Dict[str, Dict[int, float]]
    :param out_file_path: path to output file for generating tex table, ignored if not present
    :type out_file_path: str
    """
    global metricsLT

    ms = d.keys()
    pairs = [i + j for i in ms for j in ms if i < j]
    results = {}

    for p in pairs:
        results[p] = numpy.corrcoef(d[p[0]].values(), d[p[1]].values())[0, 1]

    if out_file_path:
        try:
            with open(out_file_path, 'w') as outf:
                outf.write('\\begin{tabu} to \\textwidth { | X[l] | ')

                for i, m in enumerate(d.keys()):
                    outf.write('X[l] | ')
                outf.write('}\n\\hline\n')

                outf.write('& ')
                for i, metric in enumerate(d.keys()):
                    outf.write('\\textbf{' + metricsLT[metric].capitalize() + '}')
                    if i != len(d) - 1:
                        outf.write(' & ')
                    else:
                        outf.write(' \\\\ \\hline \n')

                for i, metric in enumerate(d.keys()):
                    if i != len(d.keys()) - 1:
                        outf.write(metricsLT[metric].capitalize() + ' & ')
                        for j in range(i + 1):
                            outf.write('& ')
                        l = filter(lambda x: x.startswith(metric), results)
                        for item, key in enumerate(l):
                            outf.write(str('%.2f' % results[key]))
                            if item != len(l) - 1:
                                outf.write(' & ')
                            else:
                                outf.write(' \\\\ \\hline \n')

                outf.write('\\hline \n\\end{tabu}\n')
        except Exception as e:
            logging.critical(e)
            exit(1)

    return results


def find_elist_files(cwd):
    # type: (str) -> Dict[str, Dict[None, None]]
    """
    :param cwd: current working directory
    :return: A dictionary where the keys are the names of the files with .elist extension
    :rtype: Dict[str, Dict[]]
    """
    files = {} # type: Dict[str, Dict[None, None]]
    for afile in os.listdir(cwd):
        if afile.endswith('.elist'):
            files[os.path.splitext(afile)[0]] = {}  # a dict of <metric>, <a dict of <nodeid>, <value>>
    return files


def gen_degree_dist(G):
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(G, DegToCntV)
    return dict([(item.GetVal1(), item.GetVal2()) for item in DegToCntV])


def gen_er_graph(n_nodes, p):
    # type: (int, float) -> TUNGraph
    m_edges = int((n_nodes*(n_nodes-1))/2 * p)
    Rnd = snap.TRnd(1, 0)
    return snap.GenRndGnm(snap.PUNGraph, n_nodes, m_edges, False, Rnd)


def gen_ws_graph(n_nodes, k_neighbors, p):
    # type: (int, int, float) -> TUNGraph
    Rnd = snap.TRnd(1, 0)
    return snap.GenSmallWorld(n_nodes, k_neighbors, p, Rnd)


def gen_ba_graph(n_nodes, k_neighbors):
    Rnd = snap.TRnd(1, 0)
    return snap.GenPrefAttach(n_nodes, k_neighbors, Rnd)


def get_graph_properties(G):
    # type: (TUNGraph) -> Dict[str, Any]
    props = OrderedDict()
    props['nodes_count'] = len(list(G.Nodes()))
    props['edges_count'] = len(list(G.Edges()))
    props['triads_count'] = snap.GetTriads(G, -1)

    res = snap.TIntV()
    snap.GetDegSeqV(G, res)
    props['mean_degree'] = sum(res) / props['nodes_count']

    # take 10% of nodes, and calculate average of shortest path between them
    randNums = [random.randrange(1, props['nodes_count']) for _ in range(int(props['nodes_count'] * 0.1))]
    r = random.randrange(1, props['nodes_count'])
    while r in randNums:
        r = random.randrange(1, props['nodes_count'])
    nodesList = [NI.GetId() for NI in G.Nodes()]
    avgShortPath = sum([snap.GetShortPath(G, nodesList[i-1], nodesList[r-1]) for i in randNums]) / len(randNums)
    props['avg_sample_path_len'] = avgShortPath

    props['eff_diameter_10%'] = snap.GetBfsEffDiam(G, int(props['nodes_count']* 0.1))
    props['avg_clust_coef'] = snap.GetClustCf(G, -1)

    return props


# output is the same dict but sorted
def gen_rank_files(cwd, files):
    # type: (str, Dict[str, Dict[str, Dict[int, float]]]) -> Dict
    """
    :param files: a dictionary <filename, <metric, <nid, value>>>
    :return: A dictionary <metric, <metric, <nodeid, value>>> where dictionary <nodeid, value> has been
    sorted (desc) by value
    :rtype: Dict[str, Dict[str, Dict[int, float]]]
    """

    global metricsLT

    for fn in files.keys():
        for k in files[fn].keys():
            # first, sort them
            Dsorted = OrderedDict(sorted(files[fn][k].items(), key=lambda d: d[1], reverse=True))
            # now write them to a file
            try:
                with open(os.path.join(cwd, fn + '.' + metricsLT[k] + '.rank'), 'w') as outfile:
                    for nid, val in Dsorted.iteritems():
                        outfile.write(str(nid) + ' ' + str(val) + '\n')
            except Exception as e:
                logging.critical(str(e))
                exit(1)

            files[fn][k] = Dsorted

    return files


# input: d, a dictionary <metric, <nid, value>> where nids are ordered by value (desc)
def gen_metrics_table(file_path, d):
    # type: (str, Dict[str, Dict[str, str]]) -> None
    global metricsLT

    ids = {}
    names = {}
    for metric in d.keys():
        ids[metric] = d[metric].keys()[:10]
        names[metric] = lookup_names(ids[metric], file_path + '.names') # type: ignore #TODO check
        
    metric0 = ''
    try:
        with open(file_path + '.metrics.tex', 'w') as outf:
            outf.write('\\begin{tabu} to \\textwidth { | ')
            
            for m in d.keys():
                outf.write('X[l] | ')
            
            outf.write('}\n\\hline\n')
            
            for i, metric in enumerate(sorted(d.keys())):
                outf.write('\\textbf{' + metricsLT[metric].capitalize() + '}')
                if i == 0:
                    metric0 = metric

                if i != len(d) - 1:
                    outf.write(' & ')
                else:
                    outf.write(' \\\\ \\hline \n')

            for i in range(len(names[metric0])):
                for j, metric in enumerate(sorted(d.keys())):
                    outf.write(names[metric][i])
                    if j != len(d) - 1:
                        outf.write(' & ')
                    else:
                        outf.write(' \\\\ \\hline \n')
            outf.write('\n\\end{tabu}\n')
    except Exception as e:
        logging.critical(str(e))
        exit(1)


def get_degree(G):
    """
    Calculates the degree distribution of a graph
    :param G:
    :return: a dictionary <nodeid, value>
    """
    d = {}

    for n in G.Nodes():
        nid = n.GetId()
        d[nid] = snap.GetDegreeCentr(G, nid)

    return d


def get_closeness(G):
    """
    Calculates the closeness distribution of a graph
    :param G:
    :return: a dictionary <nodeid, value>
    """
    d = {}

    # get only the strongest connected component
    # G = snap.GetMxScc(G)

    for n in G.Nodes():
        nid = n.GetId()
        d[nid] = snap.GetClosenessCentr(G, nid)

    return d


def get_harmonic_closeness(G):
    """
    Calculates the harmonic closeness distribution of a graph
    :param G:
    :return: a dictionary <nodeid, value>
    """
    d = {}
    HDist = snap.TIntH()
    # get only the strongest connected component
    # G = snap.GetMxScc(G)  # get strongest connected component

    for n in G.Nodes():
        shPath = snap.GetShortPath(G, n.GetId(), HDist)
        # HDist maps node id to shortest path distance. Only contains nodes that are reachable from n.GetId()
        harmonic = 0
        for k in HDist:
            if HDist[k] != 0:
                harmonic += 1.0 / HDist[k]  ## wikipedia(1)
                #harmonic += 1 / (2 ** HDist[k])  ## wikipedia(2)
        d[n.GetId()] = 1.0 / (len(list(G.Nodes()))-1) * harmonic ## book
        #d[n.GetId()] = harmonic

    return d


def get_betweenness(G):
    """
    Calculates the betweenness distribution of a graph
    :param G:
    :return: a dictionary <nodeid, value>
    """
    d = {}
    BNodes = snap.TIntFltH()
    BEdges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(G, BNodes, BEdges, 1.0)

    for n in G.Nodes():
        nid = n.GetId()
        d[nid] = BNodes(nid)

    return d


def get_clustering(G):
    """
    Calculates the local clustering coefficient distribution of a graph
    :param G:
    :return: a dictionary <nodeid, value>
    """
    d = {}

    for n in G.Nodes():
        nid = n.GetId()
        d[nid] = snap.GetNodeClustCf(G, nid)

    return d


def lookup_names(xs, names_file_path):
    # type: (Optional[str], str) -> List[str]
    """
    Takes a list of names and finds their definitions in a dictionary file
    :param xs:
    :param names_file_path:
    :return:
    """
    if not os.path.exists(names_file_path):
        logging.error('File does not exist')
        return map(str, xs)

    d = {}
    try:
        with open(names_file_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if line != '' and not line.startswith('#'):
                    k, v = shlex.split(line)[0:2]
                    d[int(k)] = v

    except Exception as e:
        logging.critical(str(e))
        exit(1)

    ys = []
    for x in xs:
        ys.append(d[x])  # type: ignore

    return ys


def parse_dict_file(file_path, typ1, typ2):
    """
    Reads a dictionary stored in a file and converts keys to typ1 and values to typ2
    :param file_path: dictionary file
    :param typ1: type of keys
    :param typ2: type of values
    :return: an ordered dictionary
    """
    d = read_dict_file(file_path)
    return OrderedDict([(typ1(k),typ2(v[0])) for (k,v) in d.iteritems()])


def parse_elist_file(file_path, isDirected):
    # type: (str) -> Any
    """
    :param file_path: path to elist file
    :return: A snap undirected graph
    :rtype: snap.TUNGraph
    """

    if isDirected:
        G = snap.TNGraph.New()
    else:
        G = snap.TUNGraph.New()
    seen = set() # type: ignore #TODO check
    try:
        with open(file_path, 'r') as infile:
            for line in infile:
                line = line.strip()
                if not line.startswith('#'):
                    u, v = map(int, line.split()[0:2])  # ass. input is 2-column format
                    if u not in seen:
                        G.AddNode(u)
                        seen.update([u])
                    if v not in seen:
                        G.AddNode(v)
                        seen.update([v])
                    G.AddEdge(u, v)
    except Exception as e:
        logging.critical(str(e))
        exit(1)
    return G


def plot_degree_dist(G, gname):
    # type: (TUNGraph, str) -> str
    snap.PlotOutDegDistr(G, gname, "Degree")
    return gname + '.png'


def plot_dist(file_path, title='', xlabel='', ylabel='', xstart='', ystart='', xend='', yend='',
              ylog=False, xlog=False, varwidthcol=-1, linespoint=False):
    """
    Plots a .tab file
    :param yticks:
    :param file_path: path to .tab file
    :param gname: Graph title
    :param xaxis: X-axis title
    :param yaxis: Y-axis title
    """
    try:
        fpath, ext = os.path.splitext(file_path)
        #create plt file
        with open(fpath + '.plt', 'w') as of:
           of.write('set title "%s"\n' % title)
           of.write('set key bottom right\n')
           of.write('set grid\n')
           of.write('set xlabel "%s"\n' % xlabel)
           of.write('set ylabel "%s"\n' % ylabel)
           of.write('set terminal png size 1000,800\n')
           of.write('set output "%s.png"\n' % fpath)
           of.write('set style fill\n')

           if xlog:
               of.write('set logscale x\n')

           if ystart or yend:
               of.write('set yrange [%s:%s]\n' % (ystart, yend))

           if xstart or xend:
               of.write('set xrange [%s:%s]\n' % (xstart, xend))

           if ylog:
               of.write('set logscale y\n')
               of.write('set format y "10^{%L}"\n')
           else:
               of.write('set yrange [%s:%s]\n' % (ystart, yend))

           if linespoint:
               of.write('plot "%s.tab" using 1:2 title "" with linespoints pt 6\n' % fpath)
           elif varwidthcol == -1:
               of.write('plot "%s.tab" using 1:2 title "" with boxes\n' % fpath)
           else:
               of.write('plot "%s.tab" using 1:2:%d title "" with boxes\n' % (fpath, varwidthcol))

        #run gnuplot
        call(['gnuplot', fpath + '.plt'])

    except Exception as e:
        logging.critical(str(e))
        exit(1)

    return None


def plot_metric(d, path, fn, metric):
    """
    :param path: a path for output
    :param fn: file name
    :param metric: the metric being output
    """
    try:
        file_path = os.path.join(path, fn + '.' + metric)

        with open(file_path + '.tab', 'w') as of:
            for i, v in enumerate(d.values()):
                of.write(str(i+1) + ' ' + str(v) + '\n')


        #create plt file
        with open(file_path + '.plt', 'w') as of:
           of.write('set title "Rank vs. ' + metric.capitalize() + ' plot for ' + fn + ' dataset"\n')
           of.write('set key bottom right\n')
           of.write('set logscale x 10\n')
           of.write('set format x "10^{%L}"\n')
           of.write('set mytics 10\n')
           of.write('set grid\n')
           of.write('set xlabel "Rank"\n')
           of.write('set ylabel "' + metric.capitalize() + '"\n')
           of.write('set tics scale 2\n')
           of.write('set terminal png size 1000,800\n')
           of.write('set output \'' + file_path + '.png\'\n')
           of.write('plot    "' + file_path + '.tab" using 1:2 title "" with linespoints pt 6\n')

        #run gnuplot
        call(['gnuplot', file_path + '.plt'])

    except Exception as e:
        logging.critical(e)
        exit(1)


def rank_dict(d):
    return OrderedDict(sorted(d.items(), key=lambda d: d[1], reverse=True))


def read_dict_file(file_path):
    # type: (str) -> Union[Exception, Dict[str, List[str]]]
    d = OrderedDict() # type: ignore #TODO check
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip().startswith('#'):
                    lineA = line.rstrip().split()
                    d[lineA[0]] = lineA[1:]
    except Exception as e:
        logging.critical(str(e))
        exit(1)

    return d


def write_dict(d, file_path, header=''):
    try:
        with open(file_path, 'w') as f:
            f.write(header)
            for k, v in d.iteritems():
                f.write(str(k) + ' ' + str(v) + '\n')
    except Exception as e:
        logging.critical(str(e))


def get_global_clustering_coefficient(G):
    ct = get_closed_triads(G)
    return float(ct) / (ct + get_open_triads(G))


def get_open_triads(G):
    # get edges in tuple format
    edges = set()
    for e in G.Edges():
        edges.update((e.GetSrcNId(), e.GetDstNId()))

    # first, calculate number of open triads
    openTriads = 0
    seen = set()
    for u in G.Nodes():
        uid = u.GetId()
        NIdV = snap.TIntV()
        snap.GetNodesAtHop(G, uid, 2, NIdV, False)
        for vid in NIdV:
            if (uid, vid) not in seen and (vid, uid) not in seen:
                seen.update((uid, vid))
                if (uid, vid) not in edges and (vid, uid) not in edges:
                    openTriads += snap.GetCmnNbrs(G, uid, vid)
    return openTriads / 2


def get_closed_triads(G):
    return snap.GetTriads(G)


def write_dict_of_list_values(d, file_path, header=''):
    try:
        with open(file_path, 'w') as f:
            f.write(header)
            for k, v in d.iteritems():
                f.write(str(k) + ' ')
                for x in v:
                    f.write(str(x) + ' ')
                f.write('\n')
    except Exception as e:
        logging.critical(str(e))


def write_elist(G, file_path):
    # type: (TUNGraph, str) -> None
    """
    Takes a graph G and writes an edge list to a given file_path.
    :param G: a snap graph
    :param file_path: path to output file
    """
    try:
        with open(file_path, 'w') as f:
            for edge in G.Edges():
                f.write(str(edge.GetSrcNId()) + ' ' + str(edge.GetDstNId()) + '\n')
    except Exception as e:
        logging.critical(str(e))

