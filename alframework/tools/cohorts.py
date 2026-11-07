"""In-memory logical cohorts for pooled top-K acquisition.

A cohort is a fixed group of ``cohort_size`` trajectories. Structures are
admitted one at a time as builders produce them, so a cohort larger than the
sampler capacity still progresses. Each cohort is pinned to the model
generation that was current when it opened. After a promotion the open cohort
keeps admitting, and its remaining members still run with the pinned model,
until it is full; the next cohort opens on the then-current model. This keeps
full admission without mixing generations within a cohort.

When every member of a full cohort has a terminal outcome (``eligible``,
``completed``, ``rejected``, or ``failed``), its selector releases ``k``
candidates for labeling: the best-scoring by default, or a diverse set with
``fps``. The selection depends only on the candidates themselves, not on how
members were grouped into tasks or the order in which tasks finished. Candidates returned before a trajectory
failed remain in the pool.

There is no admission timeout and no partial-cohort selection. Cohorts are
held in memory only; unfinished cohorts are lost if the coordinator restarts.
"""

from collections import Counter, deque

from ase import Atoms

from alframework.samplers.acquisition import select_candidates
from alframework.tools.molecules_class import MoleculesObject

TERMINAL_OUTCOMES = ("eligible", "completed", "rejected", "failed")
# Finished-cohort summaries hold every selected frame ID, so long runs keep only the latest.
FINISHED_HISTORY = 100


class _Cohort:
    def __init__(self, cohort_id, model_id, selected_state=None):
        self.cohort_id = cohort_id
        self.model_id = model_id
        self.selected_state = selected_state
        self.members = []
        self.outcomes = {}
        self.candidates = []


class CohortPool:
    """Admit trajectories into cohorts and select each cohort's top ``k`` frames.

    Args:
        selection (dict): ``cohort_size`` and ``k``, as validated by
            ``acquisition.acquisition_settings``.
    """

    def __init__(self, selection):
        self.selection = dict(selection)
        self.cohort_size = int(selection["cohort_size"])
        self.k = int(selection["k"])
        self.group_by_state = bool(selection.get("group_by_selected_state", False))
        self._cohorts = {}
        self._open = {}
        self._next = {}
        self._membership = {}
        self.finished = deque(maxlen=FINISHED_HISTORY)
        self._finished_total = 0
        self._selected_total = 0

    def admit(self, molecule, model_id):
        """Assign ``molecule`` to a cohort; return the model generation it must use."""
        trajectory_id = molecule.get_moleculeid()
        if trajectory_id in self._membership:
            raise ValueError(f"Trajectory {trajectory_id!r} was already admitted to a cohort.")
        state = molecule.get_metadata().get('selected_state') if self.group_by_state else None
        if self.group_by_state and (isinstance(state, bool) or not isinstance(state, int) or state < 0):
            raise ValueError('State-grouped admission requires resolved selected_state metadata')
        cohort = self._open.get(state)
        if cohort is None or len(cohort.members) >= self.cohort_size:
            ordinal = self._next.get(model_id, 0)
            self._next[model_id] = ordinal + 1
            cohort = _Cohort("cohort-{:04d}-{:06d}".format(int(model_id), ordinal), model_id, state)
            self._cohorts[cohort.cohort_id] = cohort
            self._open[state] = cohort
        cohort.members.append(trajectory_id)
        self._membership[trajectory_id] = cohort.cohort_id
        molecule.update_metadata({"cohort_id": cohort.cohort_id,
                                  "trajectory_ordinal": len(cohort.members) - 1,
                                  "cohort_model_id": cohort.model_id})
        return cohort.model_id

    def record(self, molecule):
        """Record one returned trajectory's terminal outcome and candidates."""
        cohort, trajectory_id = self._member(molecule.get_moleculeid())
        if trajectory_id in cohort.outcomes:
            raise ValueError(f"Trajectory {trajectory_id!r} already reported to {cohort.cohort_id}.")
        metadata = molecule.get_metadata()
        if self.group_by_state:
            if metadata.get('selected_state') != cohort.selected_state:
                raise ValueError('Trajectory changed its selected state within a state-specific cohort')
            if any(c.get('metadata', {}).get('selected_state') != cohort.selected_state
                   for c in metadata.get('acquisition_candidates', [])):
                raise ValueError('Candidate state disagrees with its cohort')
        outcome = metadata.get("trajectory_outcome", "failed")
        if outcome not in TERMINAL_OUTCOMES:
            raise ValueError(f"Trajectory {trajectory_id!r} reported non-terminal outcome {outcome!r}.")
        cohort.outcomes[trajectory_id] = outcome
        cohort.candidates.extend(metadata.get("acquisition_candidates", []))

    def record_failed(self, molecules):
        """Mark members of a task that failed as a whole; they returned no candidates."""
        for molecule in molecules:
            cohort, trajectory_id = self._member(molecule.get_moleculeid())
            cohort.outcomes.setdefault(trajectory_id, "failed")

    def _member(self, trajectory_id):
        if trajectory_id not in self._membership:
            raise ValueError(f"Trajectory {trajectory_id!r} was not admitted to a cohort.")
        return self._cohorts[self._membership[trajectory_id]], trajectory_id

    def ready(self):
        """Finalize every complete cohort; return the selected frames as ``MoleculesObject`` instances."""
        selected = []
        for cohort_id in list(self._cohorts):
            cohort = self._cohorts[cohort_id]
            if len(cohort.members) < self.cohort_size or len(cohort.outcomes) < len(cohort.members):
                continue
            chosen = select_candidates(cohort.candidates, self.selection)
            for rank, candidate in enumerate(chosen):
                selected.append(self._frame(cohort, candidate, rank))
            self.finished.append({
                "cohort_id": cohort.cohort_id, "model_id": cohort.model_id, "members": len(cohort.members),
                "selected_state": cohort.selected_state,
                "outcomes": dict(Counter(cohort.outcomes.values())), "candidates": len(cohort.candidates),
                "selected": [candidate["frame_id"] for candidate in chosen]})
            self._finished_total += 1
            self._selected_total += len(chosen)
            for trajectory_id in cohort.members:
                del self._membership[trajectory_id]
            del self._cohorts[cohort_id]
            if self._open.get(cohort.selected_state) is cohort:
                del self._open[cohort.selected_state]
        return selected

    def _frame(self, cohort, candidate, rank):
        atoms = Atoms(numbers=candidate["numbers"], positions=candidate["positions"],
                      cell=candidate["cell"], pbc=candidate["pbc"])
        if candidate.get('topology_atom_ids') is not None:
            from alframework.tools.atom_order import validated_atom_ids
            atoms.set_array('topology_atom_ids', validated_atom_ids(candidate['topology_atom_ids'], len(atoms)))
        frame = MoleculesObject(atoms, "{}-s{:09d}".format(candidate["trajectory_id"], candidate["step"]))
        frame.update_metadata({**candidate.get('metadata', {}),
                               "cohort_id": cohort.cohort_id, "cohort_model_id": cohort.model_id,
                               "frame_id": candidate["frame_id"], "trajectory_id": candidate["trajectory_id"],
                               "step": candidate["step"], "time_ps": candidate["time_ps"],
                               "ranking_score": candidate["ranking_score"], "cohort_rank": rank,
                               "cohort_candidates": len(cohort.candidates), **candidate["signals"],
                               **candidate["flags"]})
        return frame

    def status(self):
        """JSON-safe summary of open and finished cohorts."""
        open_cohorts = [{"cohort_id": cohort.cohort_id, "model_id": cohort.model_id,
                         "admitted": len(cohort.members), "finished": len(cohort.outcomes),
                         **({"selected_state": cohort.selected_state} if self.group_by_state else {}),
                         "candidates": len(cohort.candidates)} for cohort in self._cohorts.values()]
        return {"cohort_size": self.cohort_size, "k": self.k, "open": open_cohorts,
                "finished": self._finished_total, "selected": self._selected_total}
