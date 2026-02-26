"""
A* assembler for puzzle fragment reconstruction.

Uses A* search (Best-first search with admissible heuristic) to find
an assembly ordering that maximises total compatibility score.

State:  (frozenset of placed fragment ids, dict of placements)
g(s):   sum of edge scores for all matched edge pairs so far (negated, since A* minimises)
h(s):   admissible upper-bound estimate of remaining achievable score
        = sum of best possible scores for unplaced fragment best edges

The search is bounded by:
  - max_states: maximum number of states to expand (beam/budget limit)
  - beam_width: number of states kept in the open set at each step

For small N (≤ 8 fragments) the search is near-exhaustive.
For larger N it degrades gracefully to a best-first greedy heuristic.
"""
from __future__ import annotations

import heapq
import numpy as np
from typing import Dict, FrozenSet, List, Optional, Tuple

from ..models import Fragment, CompatEntry, Assembly


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_edge_to_frag(fragments: List[Fragment]) -> Dict[int, Fragment]:
    """Map edge_id → Fragment."""
    mapping: Dict[int, Fragment] = {}
    for frag in fragments:
        for edge in frag.edges:
            mapping[edge.edge_id] = frag
    return mapping


def _build_best_score_per_frag(
    fragments: List[Fragment],
    entries: List[CompatEntry],
    edge_to_frag: Dict[int, Fragment],
) -> Dict[int, float]:
    """
    For each fragment, compute its best reachable compat score (upper bound).
    Used for the admissible heuristic h(s).
    """
    best: Dict[int, float] = {f.fragment_id: 0.0 for f in fragments}
    for e in entries:
        fi = edge_to_frag.get(getattr(e.edge_i, "edge_id", -1))
        fj = edge_to_frag.get(getattr(e.edge_j, "edge_id", -1))
        if fi is None or fj is None:
            continue
        best[fi.fragment_id] = max(best[fi.fragment_id], e.score)
        best[fj.fragment_id] = max(best[fj.fragment_id], e.score)
    return best


def _score_for_placement(
    frag: Fragment,
    placed_ids: FrozenSet[int],
    entries: List[CompatEntry],
    edge_to_frag: Dict[int, Fragment],
) -> float:
    """
    Compute the incremental score gained by placing *frag* given the already
    placed set *placed_ids*.  Returns the sum of scores for all entry pairs
    where one fragment is *frag* and the other is already placed.
    """
    score = 0.0
    for e in entries:
        fi = edge_to_frag.get(getattr(e.edge_i, "edge_id", -1))
        fj = edge_to_frag.get(getattr(e.edge_j, "edge_id", -1))
        if fi is None or fj is None:
            continue
        i_id = fi.fragment_id
        j_id = fj.fragment_id
        if i_id == frag.fragment_id and j_id in placed_ids:
            score += e.score
        elif j_id == frag.fragment_id and i_id in placed_ids:
            score += e.score
    return score


def _heuristic(
    unplaced: FrozenSet[int],
    best_per_frag: Dict[int, float],
) -> float:
    """
    Admissible upper bound: sum of best possible edge scores for unplaced frags.
    """
    return sum(best_per_frag.get(fid, 0.0) for fid in unplaced)


def _place_new_fragment(
    new_frag: Fragment,
    anchor_frag_id: int,
    placements: Dict[int, Tuple[np.ndarray, float]],
    best_entry: Optional[CompatEntry],
    frag_index: int,
) -> Tuple[np.ndarray, float]:
    """
    Compute a simple placement for *new_frag* relative to an anchor fragment.
    Places fragments in a grid pattern offset by (frag_index * 100, 0).
    """
    if anchor_frag_id in placements:
        anchor_pos, anchor_rot = placements[anchor_frag_id]
        # Simple offset: place next to the anchor
        offset = np.array([float(frag_index) * 120.0, 0.0])
        return anchor_pos + offset, anchor_rot
    return np.array([float(frag_index) * 120.0, 0.0]), 0.0


# ---------------------------------------------------------------------------
# A* state
# ---------------------------------------------------------------------------

class _AStarState:
    """A single state in the A* search."""

    __slots__ = ("placed_ids", "placements", "g_score", "f_score")

    def __init__(
        self,
        placed_ids: FrozenSet[int],
        placements: Dict[int, Tuple[np.ndarray, float]],
        g_score: float,
        h_score: float,
    ) -> None:
        self.placed_ids  = placed_ids
        self.placements  = placements
        self.g_score     = g_score
        self.f_score     = -(g_score + h_score)  # negated: heap is min-heap, we maximise

    def __lt__(self, other: "_AStarState") -> bool:
        return self.f_score < other.f_score


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def astar_assembly(
    fragments: List[Fragment],
    entries: List[CompatEntry],
    max_states: int = 10_000,
    beam_width: int = 50,
) -> Assembly:
    """
    A*-based puzzle assembly.

    Args:
        fragments:   All fragments to be assembled.
        entries:     Compatibility entries, sorted descending by score.
        max_states:  Maximum number of states to expand before returning best.
        beam_width:  Maximum number of states in the open set (beam pruning).

    Returns:
        Assembly with placements and total_score.
    """
    if not fragments:
        return Assembly(fragments=fragments, placements={},
                        compat_matrix=np.array([]), method="astar")

    edge_to_frag   = _build_edge_to_frag(fragments)
    best_per_frag  = _build_best_score_per_frag(fragments, entries, edge_to_frag)
    all_frag_ids   = frozenset(f.fragment_id for f in fragments)
    frag_by_id     = {f.fragment_id: f for f in fragments}

    # Initial state: place first fragment at origin
    first = fragments[0]
    init_placed    = frozenset([first.fragment_id])
    init_placement = {first.fragment_id: (np.array([0.0, 0.0]), 0.0)}
    init_h = _heuristic(all_frag_ids - init_placed, best_per_frag)
    init_state = _AStarState(init_placed, init_placement, 0.0, init_h)

    open_heap: List[_AStarState] = [init_state]
    heapq.heapify(open_heap)

    best_complete: Optional[_AStarState] = None
    best_partial:  Optional[_AStarState] = init_state
    states_expanded = 0

    while open_heap and states_expanded < max_states:
        state = heapq.heappop(open_heap)
        states_expanded += 1

        # Update best partial
        if (best_partial is None or
                len(state.placed_ids) > len(best_partial.placed_ids) or
                (len(state.placed_ids) == len(best_partial.placed_ids) and
                 state.g_score > best_partial.g_score)):
            best_partial = state

        # Goal check: all fragments placed
        if state.placed_ids == all_frag_ids:
            if best_complete is None or state.g_score > best_complete.g_score:
                best_complete = state
            continue

        # Expand: try adding each unplaced fragment
        unplaced = all_frag_ids - state.placed_ids
        for fid in sorted(unplaced):
            frag = frag_by_id[fid]
            incr = _score_for_placement(frag, state.placed_ids, entries, edge_to_frag)
            new_g = state.g_score + incr

            new_placed = state.placed_ids | frozenset([fid])
            new_h      = _heuristic(all_frag_ids - new_placed, best_per_frag)

            # Find anchor: any already-placed fragment that has an entry with frag
            anchor_id = next(iter(state.placed_ids))
            for e in entries:
                fi = edge_to_frag.get(getattr(e.edge_i, "edge_id", -1))
                fj = edge_to_frag.get(getattr(e.edge_j, "edge_id", -1))
                if fi is None or fj is None:
                    continue
                if fi.fragment_id == fid and fj.fragment_id in state.placed_ids:
                    anchor_id = fj.fragment_id
                    break
                if fj.fragment_id == fid and fi.fragment_id in state.placed_ids:
                    anchor_id = fi.fragment_id
                    break

            new_pos, new_rot = _place_new_fragment(
                frag, anchor_id, state.placements, None, len(state.placed_ids)
            )
            new_placements = dict(state.placements)
            new_placements[fid] = (new_pos, new_rot)

            child = _AStarState(new_placed, new_placements, new_g, new_h)

            heapq.heappush(open_heap, child)
            # Beam pruning: keep only beam_width best
            if len(open_heap) > beam_width:
                # Remove the worst (highest f_score = least promising) state
                # We use a workaround: sort and truncate
                open_heap.sort()
                open_heap = open_heap[:beam_width]
                heapq.heapify(open_heap)

    # Return best result found
    result_state = best_complete if best_complete is not None else best_partial

    if result_state is None:
        return Assembly(fragments=fragments, placements={},
                        compat_matrix=np.array([]), method="astar",
                        total_score=0.0)

    return Assembly(
        fragments=fragments,
        placements=result_state.placements,
        compat_matrix=np.array([]),
        method="astar",
        total_score=float(result_state.g_score),
    )
