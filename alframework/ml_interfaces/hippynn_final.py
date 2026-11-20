"""Bounded final evaluation for ALF's HIPPYNN loss and plot graphs."""

import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from hippynn import settings
from hippynn.graphs import GraphModule
from hippynn.graphs.indextypes import IdxType
from hippynn.graphs.nodes import indexers, loss
# The algebra nodes are defined in base.algebra; newer HIPPYNN also re-exports them from base, released 0.1.3 doesn't.
from hippynn.graphs.nodes.base import IndexNode, algebra
from hippynn.plotting import Hist2D, PlotMaker, SensitivityPlot


ROW_NODES = (algebra.AtLeast2D, indexers.OneHotEncoder, indexers.PaddingIndexer, indexers.AtomReIndexer, IndexNode)
SCALAR_NODES = (algebra.ValueNode, algebra.AddNode, algebra.SubNode, algebra.MulNode, algebra.DivNode,
                algebra.PowNode, algebra.NegNode)
REDUCERS = (loss.MAELoss, loss.MSELoss, loss.Mean)


def _index_state(node):
    """A node's index state: ``index_state`` in newer HIPPYNN, ``_index_state`` in released 0.1.3."""
    try:
        return node.index_state
    except (AttributeError, ValueError):  # HIPPYNN looks unknown attributes up among the node's inputs
        return getattr(node, '_index_state', None)


def _operations(graph):
    return [(node, graph.get_module(node)) for node in graph.forward_output_list]


def _run(operations, values):
    for node, module in operations:
        values[node] = module(*(values[parent] for parent in node.parents))
    return values


def _reduce(node, args):
    if type(node) is loss.Mean:
        values, = args
    else:
        values = args[0] - args[1]
        values = values.abs() if type(node) is loss.MAELoss else values.square()
    return values.sum(dtype=torch.float64).item(), values.numel(), values.dtype


def _combine(reductions):
    present = [entry for entry in reductions if entry is not None]
    if not present:
        return None
    return sum(entry[0] for entry in present), sum(entry[1] for entry in present), present[0][2]


class FinalEvaluation:
    """Check the graph before using per-batch reductions. Custom graphs use the existing evaluator."""

    @classmethod
    def from_evaluator(cls, evaluator):
        graph = evaluator.loss
        if type(graph) is not GraphModule or settings.DEBUG_LOSS_BROADCAST:
            return None
        scalar = {node for node in graph.input_nodes if _index_state(node) is IdxType.Scalar}
        rows = set(graph.input_nodes) - scalar
        row_ops, reductions, scalar_ops = [], [], []
        for node, module in _operations(graph):
            parents = set(node.parents)
            if type(node) in ROW_NODES and parents <= rows:
                row_ops.append((node, module))
                rows.add(node)
            elif type(node) in REDUCERS and parents <= rows:
                reductions.append(node)
                scalar.add(node)
            elif type(node) in SCALAR_NODES and parents <= scalar:
                scalar_ops.append((node, module))
                scalar.add(node)
            else:
                return None
        if not set(graph.nodes_to_compute) <= scalar:
            return None

        plotter = evaluator.plot_maker
        histograms = []
        if plotter:
            if type(plotter) is not PlotMaker or type(plotter.torch_module) is not GraphModule:
                return None
            column = 0
            for plot in plotter.plotters:
                if type(plot) is Hist2D and type(plot.bins) is int and plot.bins > 0:
                    if plot.saved or plot.shown:
                        histograms.append((plot, column))
                elif type(plot) is not SensitivityPlot or plot.parents:
                    return None
                column += len(plot.parents)
            if any(type(node) not in ROW_NODES for node in plotter.torch_module.forward_output_list):
                return None

        return cls(evaluator, row_ops, reductions, scalar_ops, histograms)

    def __init__(self, evaluator, row_ops, reductions, scalar_ops, histograms):
        self.evaluator = evaluator
        self.input_nodes = evaluator.loss.input_nodes
        self.scalar_columns = [i for i, node in enumerate(self.input_nodes) if _index_state(node) is IdxType.Scalar]
        self.row_ops = row_ops
        self.reductions = reductions
        self.scalar_ops = scalar_ops
        self.histograms = histograms

    @torch.no_grad()
    def batch(self, inputs):
        values = _run(self.row_ops, dict(zip(self.input_nodes, inputs)))
        parts = [_reduce(node, [values[parent] for parent in node.parents]) for node in self.reductions]
        plots = []
        if self.histograms:
            columns = self.evaluator.plot_maker.torch_module(*inputs)
            plots = [(columns[i].detach().numpy().reshape(-1), columns[i + 1].detach().numpy().reshape(-1))
                     for _, i in self.histograms]
        return parts, plots

    @torch.no_grad()
    def loss_values(self, rank_summaries):
        first = rank_summaries[0]['scalars']
        if first is None:
            raise ValueError('The final evaluation split has no batches.')
        values = {self.input_nodes[i]: value for i, value in zip(self.scalar_columns, first)}
        for node, parts in zip(self.reductions, zip(*(s['reductions'] for s in rank_summaries))):
            total, count, dtype = _combine(parts)
            values[node] = torch.tensor(total / count if count else float('nan'), dtype=dtype)
        _run(self.scalar_ops, values)
        return {name: values[node].mean().item()
                for name, node in zip(self.evaluator.loss_names, self.evaluator.loss.nodes_to_compute)}

    def plot(self, name, when, edges, counts):
        maker = self.evaluator.plot_maker
        if not maker or when is None:
            return
        if isinstance(when, int):
            if when % maker.plot_every:
                return
            when = f'epochs/epoch{when}'
        if not isinstance(when, str):
            raise ValueError(f'Invalid final evaluation phase {when!r}.')

        from matplotlib import pyplot as plt

        location = Path(maker.make_full_location(os.path.join(when, name)))
        print(f'Making plots. Saved location: {location}')
        histograms = iter(zip(edges, counts))
        for plot in maker.plotters:
            if not (plot.shown or plot.saved):
                continue
            if type(plot) is Hist2D:
                (x, y, limits), counts = next(histograms)
                fig, ax = plt.subplots()
                mesh = ax.pcolormesh(x, y, counts.T, norm=plot.norm)
                ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), xlabel=plot.xlabel, ylabel=plot.ylabel)
                if plot.add_identity_line:
                    ax.plot(limits, limits, c='r', lw=0.5)
                fig.colorbar(mesh)
            else:
                fig = plot.make_plot(())
            if plot.shown:
                plt.show()
            if plot.saved:
                path = location / plot.saved
                print(f'Saving plot at {path}')
                fig.savefig(path, transparent=settings.TRANSPARENT_PLOT)
            plt.close(fig)


def _extend_ranges(old, plots):
    for i, (x, y) in enumerate(plots):
        if x.size:
            bounds = (x.min(), x.max(), y.min(), y.max())
            if old[i] is not None:
                bounds = tuple(op(a, b) for op, a, b in zip((np.minimum, np.maximum) * 2, old[i], bounds))
            old[i] = bounds


def _edges(plot, ranges):
    ranges = np.array([bounds for bounds in ranges if bounds is not None])
    if not ranges.size:
        raise ValueError('A final evaluation histogram has no values in this split.')
    x = ranges[:, :2]
    y = ranges[:, 2:]
    _, x_edges, y_edges = np.histogram2d(x.reshape(-1), y.reshape(-1), bins=plot.bins)
    return x_edges, y_edges, (min(x.min(), y.min()), max(x.max(), y.max()))


def _write(path, data):
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def _rank_evaluation(plan, splits, spool):
    summaries = {}
    for name, batches in splits:
        reductions = [None] * len(plan.reductions)
        ranges = [None] * len(plan.histograms)
        scalars = None
        for inputs in batches:
            if scalars is None:
                scalars = [inputs[i] for i in plan.scalar_columns]
                if any(value.ndim != 0 for value in scalars):
                    raise ValueError('Expected scalar predictions from the final evaluation graph.')
            parts, plots = plan.batch(inputs)
            reductions = [_combine(pair) for pair in zip(reductions, parts)]
            if plots:
                _extend_ranges(ranges, plots)
                pickle.dump((name, plots), spool, protocol=pickle.HIGHEST_PROTOCOL)
        summaries[name] = dict(scalars=scalars, reductions=reductions, ranges=ranges)
    return summaries


def _histogram_edges(plan, ranks):
    return {name: [_edges(plot, (rank[name][i] for rank in ranks))
                   for i, (plot, _) in enumerate(plan.histograms)] for name in ranks[0]}


def _histogram_counts(spool, edges):
    counts = _empty_histograms(edges)
    spool.seek(0)
    while spool.peek(1):
        name, plots = pickle.load(spool)
        for total, (x, y, _), (true, predicted) in zip(counts[name], edges[name], plots):
            histogram, _, _ = np.histogram2d(true, predicted, bins=(x, y))
            total += histogram.astype(np.int64)
    return counts


def _empty_histograms(edges):
    return {name: [np.zeros((len(x) - 1, len(y) - 1), dtype=np.int64) for x, y, _ in bounds]
            for name, bounds in edges.items()}


def _sparse_histogram(counts):
    flat = counts.ravel()
    indices = np.flatnonzero(flat)
    return indices, flat[indices]


def write_rank_summary(plan, splits, directory, rank, world):
    """Keep only plot columns on local scratch. Collect shared bounds, then send small histograms."""
    directory = Path(directory)
    local = os.environ.get('ALF_NODE_LOCAL_DIR')
    if local:
        Path(local).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=local) as spool:
        summaries = _rank_evaluation(plan, splits, spool)
        counts = {}
        if plan.histograms:
            _write(directory / f'ranges-{rank}.pkl', {name: s['ranges'] for name, s in summaries.items()})
            edge_file = directory / 'edges.pkl'
            if rank == 0:
                while len(list(directory.glob('ranges-*.pkl'))) < world:
                    time.sleep(0.5)
                ranges = [_read(directory / f'ranges-{i}.pkl') for i in range(world)]
                _write(edge_file, _histogram_edges(plan, ranges))
            else:
                while not edge_file.exists():
                    time.sleep(0.5)
            counts = _histogram_counts(spool, _read(edge_file))
        _write(directory / f'final-{rank}.pkl',
               (summaries, {name: [_sparse_histogram(h) for h in plots] for name, plots in counts.items()}))


def finish(plan, splits, directory, world, when):
    directory = Path(directory)
    edges = _read(directory / 'edges.pkl') if plan.histograms else {}
    histograms = _empty_histograms(edges)
    reports = []
    for rank in range(world):
        summaries, counts = _read(directory / f'final-{rank}.pkl')
        reports.append(summaries)
        for name, plots in counts.items():
            for total, (indices, values) in zip(histograms[name], plots):
                total.ravel()[indices] += values
    metrics = {}
    for name in splits:
        metrics[name] = plan.loss_values([summaries[name] for summaries in reports])
        if plan.evaluator.plot_maker:
            plan.plot(name, when, edges.get(name, ()), histograms.get(name, ()))
    return metrics
