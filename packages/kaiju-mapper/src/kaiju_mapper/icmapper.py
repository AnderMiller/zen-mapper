"""
An implementation of the adaptive cover described in:

  https://www.sci.utah.edu/~beiwang/publications/TDA_workshop_xmean_BeiWang_2021.pdf

IC-Mapper, introduced in Chalapathi, et al. [#icmapper]_, is an adaptive cover
scheme that optimizes the skeletonization of data obtained by embedding the
mapper graph via node centroids. The cover iteratively splits intervals when
doing so improves an information criterion (BIC or AIC) computed from the
clustering induced by nearest-node-centroid assignment from a partial
mapper complex.

The original paper outlines three choices and seems to claim that they
result in different outputs. However, it seems to me that their splitting
criterion (as implemented) is localized to nodes within the selected intervals.
So the BFS, DFS, and Randomized variants of their IC-Mapper should just
traverse the same splitting tree in different orders.

I imagine they meant to implement a less local variant with these options
in order for the splitting tree to be different. The smallest fix I see doing
this would be changing the following from an 'or' to an 'and':

  https://github.com/tdavislab/mapper-xmean-cover/blob/23cfdcc8c1da7d3be8434212456ac2e805154090/mapper_xmean_cover/graph.py#L103

Doing this would require us to compute the 1-D Mapper complex many times
which could be problematic.

Here is an implementation with an optimized best-first-split for the local
case. We also include a global implementation for the Best-First-Search,
Breadth-First-Search, Depth-First-Search, and Randomized splitting schemes.
These search methods are also available for the local criteria but will only result in
different covers if the cover scheme halts before convergence.

    .. [#icmapper] N. Chalapathi, Y. Zhou and B. Wang,
        "Adaptive Covers for Mapper Graphs Using Information Criteria,"
        2021 IEEE International Conference on Big Data (Big Data),
        Orlando, FL, USA, 2021, pp. 3789-3800, doi: 10.1109/BigData52589.2021.9671324.
"""

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from functools import cmp_to_key
from math import log, pi

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from zen_mapper import mapper, precomputed_cover
from zen_mapper.types import Clusterer, Cover

from kaiju_mapper.types import Seed

__all__ = ("ICMapperCoverScheme", "ICInterval")


_logger = logging.getLogger("kaiju_mapper")


_VALID_METHODS = {"best", "bfs", "dfs", "random"}


# TODO: Extend to n-dimensions
# TODO: I could see value in creating an extraTypes.py inside of Kaiju
#   Or inside of Zen Mapper to expose something like cover._grid as a tuple[Interval]
@dataclass(order=True, slots=True)
class ICInterval:
    """A 1-D cover interval together with the indices of its members."""

    lower_bound: float = field(compare=True)
    upper_bound: float = field(compare=False)
    members: np.ndarray | None = field(default=None, compare=False)

    @property
    def length(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def center(self):
        return (self.lower_bound + self.upper_bound) / 2

    @classmethod
    def from_center(cls, center: float, length: float) -> "ICInterval":
        """Construct interval using a center-point and length."""
        return cls(center - length / 2, center + length / 2)


@dataclass(order=True, slots=True)
class _Candidate:
    neg_gain: float  # min in heap == max gain
    interval: ICInterval = field(compare=False)
    children: tuple[ICInterval, ICInterval] | None = field(compare=False)
    cached_score: float = field(compare=False)


# TODO: Make this and G-Mapper able to refine an existing cover.
# I imagine something like being able to input the width balanced cover
# but that would involve adding support for an interval class
@dataclass
class ICMapperCoverScheme:
    data: np.ndarray
    """The original dataset"""
    lens: np.ndarray
    """The filtered (1 dimensional) dataset."""
    clusterer: Clusterer
    """The clusterer used by zen mapper for creating the node-centroid hard clustering."""
    delta: float = 0.0
    """An improvement factor needed to justify a split. With delta=0 any improvement to the
    BIC/AIC will justify a split."""
    iterations: int = 50
    """Max number of iterations. As of now I am unsure what this should be for given max_intervals is present."""
    max_intervals: int = 100
    """Maximum number of cover elements in the output."""
    overlap: float = 0.3
    """Proportion of the length of sub-intervals to their overlap:
      this results in a subcover that looks like width_balanced_cover
      with percent_overlap=overlap"""
    use_aic: bool = False
    """Use AIC instead of BIC for the purpose of determining if splitting should occur."""
    use_local: bool = False
    """Use a local version of the AIC/BIC."""
    init_cover: list[ICInterval] | None = None
    """Initial for IC mapper to start splitting on. If set to None then the initial
    cover is just a single interval containing all the datapoints."""
    method: str = "best"
    """Splitting traversal strategy. One of: 'best' (best-first), 'bfs'
    (breadth-first), 'dfs' (depth-first), 'random' (randomized, with
    probability proportional to interval length)."""
    seed: Seed | None = None

    # internal state
    intervals: list[ICInterval] = field(init=False, default_factory=list)
    _centroid_cache: dict[tuple[int, ...], np.ndarray] = field(
        init=False, default_factory=dict
    )
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError(f"iterations must be > 0, got {self.iterations}")
        if self.max_intervals < 1:
            raise ValueError(f"max_intervals must be > 0, got {self.max_intervals}")
        if not (0.0 < self.overlap < 1.0):
            raise ValueError(f"overlap value must be in (0,1), got {self.overlap}")
        if self.lens.squeeze().ndim > 1:
            raise ValueError(f"Lens must be 1-D, got shape: {self.lens.shape}")
        if len(self.data) != len(self.lens):
            raise ValueError(
                "The indices of data and lens must match. "
                f"Got len(data)={len(self.data)} vs len(lens)={len(self.lens)}"
            )
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got {self.method!r}"
            )
        self.rng = np.random.default_rng(self.seed)

    def __call__(self) -> Cover:
        if self.method == "best":
            if self.use_local:
                self.intervals = self._local_best_fs()
            else:
                self.intervals = self._global_best_fs()
        elif self.method == "bfs":
            if self.use_local:
                self.intervals = self._local_bfs()
            else:
                self.intervals = self._global_bfs()
        elif self.method == "dfs":
            if self.use_local:
                self.intervals = self._local_dfs()
            else:
                self.intervals = self._global_dfs()
        elif self.method == "random":
            if self.use_local:
                self.intervals = self._local_randomized()
            else:
                self.intervals = self._global_randomized()
        else:  # defensive; validated in __post_init__
            raise ValueError(f"Unknown method: {self.method!r}")

        # double check no duplicates -- can throw off max_intervals
        # but avoids having to call it too much
        self._remove_duplicate_cover_elements()
        return [
            interval.members
            for interval in self.intervals
            if interval.members is not None
        ]

    def _get_initial_intervals(
        self,
    ) -> list[ICInterval]:
        if not self.init_cover:
            tol = 1e-10
            low = self.lens.min() - tol
            high = self.lens.max() + tol
            members = np.where(~np.isnan(self.lens))[0]
            return [ICInterval(lower_bound=low, upper_bound=high, members=members)]

        else:
            membered_intervals: list[ICInterval] = list()
            for trvl in self.init_cover:
                trvl.members = np.where(
                    (trvl.lower_bound <= self.lens) & (self.lens <= trvl.upper_bound)
                )[0]
                membered_intervals.append(trvl)
            return membered_intervals

    def _with_members(self, interval: ICInterval) -> ICInterval:
        """Populates members of an ICInterval"""
        mask = (self.lens >= interval.lower_bound) & (self.lens <= interval.upper_bound)
        interval.members = np.flatnonzero(mask)
        return interval

    def _split(
        self,
        interval: ICInterval,
    ) -> tuple[ICInterval, ICInterval]:
        """Split an interval. Assume you have the original ICInterval members filled out."""
        if interval.members is None:
            _logger.info("Splitting an empty interval")

        # length of the two subintervals so that they overlap with proportion self.overlap
        split_length = interval.length / (2 - self.overlap)

        left = ICInterval(
            lower_bound=interval.lower_bound,
            upper_bound=interval.lower_bound + split_length,
        )

        right = ICInterval(
            lower_bound=interval.upper_bound - split_length,
            upper_bound=interval.upper_bound,
        )

        return self._with_members(left), self._with_members(right)

    def _remove_duplicate_cover_elements(self, debug=True):
        # Sort the cover elements by starting element
        intervals_list = list(x for x in self.intervals)

        # Custom comparator to use python's built in compare
        def c(a: ICInterval, b: ICInterval):
            if a.lower_bound < b.lower_bound:
                return -1
            elif a.lower_bound == b.lower_bound:
                if a.upper_bound > b.lower_bound:
                    return -1
                elif a.upper_bound < b.upper_bound:
                    return 1
                else:
                    return 0
            else:
                return 1

        intervals_list.sort(key=cmp_to_key(c))
        marked_for_deletion = []
        for i in range(len(intervals_list)):
            for j in range(i + 1, len(intervals_list)):
                a: ICInterval = intervals_list[i]
                b: ICInterval = intervals_list[j]
                # Contained case
                if a.lower_bound <= b.lower_bound and a.upper_bound >= b.upper_bound:
                    marked_for_deletion.append(b)
        if not debug:
            print(f"Deleted {len(marked_for_deletion)} intervals")
        for d in marked_for_deletion:
            if (
                d in intervals_list
            ):  # Might have duplicates due to numerical quirks - float comparison
                intervals_list.remove(d)

        self.intervals = intervals_list
        self.num_intervals = len(intervals_list)

    def _get_centroid(self, node_tuple: tuple[int, ...]) -> np.ndarray:
        # TODO: See if adding flags to cache intermediary operations for zen mapper good?
        # in this case: having zen mapper cache clustering results would be nice.
        c = self._centroid_cache.get(node_tuple)
        if c is None:
            c = np.average(self.data[list(node_tuple)], axis=0)
            self._centroid_cache[node_tuple] = c
        return c

    def _bic_aic(
        self,
        points: np.ndarray,
        nodes: list[tuple[int, ...]],
    ) -> float:
        """Compute BIC or AIC of a hard clustering given points and nodes"""

        if points.size == 0 or not nodes:
            return float("-inf")

        centroids = np.asarray([self._get_centroid(n) for n in nodes])
        assignments, distances = pairwise_distances_argmin_min(points, centroids)

        N = points.shape[0]
        d = points.shape[1]
        k = centroids.shape[0]

        # get effective k since we are working locally
        counts = np.bincount(assignments, minlength=k)
        nonempty = counts > 0
        k_effective = nonempty.sum()

        wcss = np.sum(distances**2)
        # nonzero counts
        nz = counts[nonempty]
        log_term = np.sum(nz * (np.log(nz) - np.log(N)))

        denom = N - k_effective
        if denom > 0 and wcss > 0:
            var = wcss / denom
        else:
            # effectively each point is a node
            # keep log() finite
            var = 1e-12

        t2 = -1 * (N * d / 2) * np.log(2 * pi * var)
        t3 = -0.5 * denom
        llh = log_term + t2 + t3

        num_params = k_effective - 1 + d * k_effective + 1
        # used for local vs global criterion
        global_num_params = k - 1 + d * k + 1

        if self.use_aic:
            if not self.use_local:
                return 2 * (llh - global_num_params)
            return 2 * (llh - num_params)  # aic

        # else use BIC
        if not self.use_local:
            return llh - 0.5 * global_num_params * log(
                N
            )  # bic with all clusters as params.

        return llh - 0.5 * num_params * log(N)  # bic

    def _build_induced_nodes(self, subcover: list[ICInterval]) -> list[tuple[int, ...]]:
        """Run mapper on subcover and return node membership"""
        members_list = [iv.members for iv in subcover if iv.members is not None]
        if not members_list or all(m.size == 0 for m in members_list):
            return []

        cover_scheme = precomputed_cover(members_list)
        result = mapper(
            data=self.data,
            projection=self.lens,
            cover_scheme=cover_scheme,
            clusterer=self.clusterer,
            dim=0,
        )

        return [tuple(int(i) for i in node) for node in result.nodes]

    def _score(self, subcover: list[ICInterval]) -> float:
        """IC of the subgraph induced by subcover only on local points"""
        members_list = [iv.members for iv in subcover if iv.members is not None]
        if not members_list:
            return float("-inf")
        local_idx = np.unique(np.concatenate(members_list))
        if local_idx.size == 0:
            return float("-inf")
        nodes = self._build_induced_nodes(subcover)
        if not nodes:
            return float("-inf")
        return self._bic_aic(self.data[local_idx], nodes)

    def _evaluate(self, interval: ICInterval) -> _Candidate:
        # only used for local since cache would die globally
        cached = self._score([interval])
        if not np.isfinite(cached):
            return _Candidate(np.inf, interval, None, cached)
        children = self._split(interval)
        new = self._score(list(children))
        if not np.isfinite(new):
            return _Candidate(np.inf, interval, None, cached)
        gain = new - cached
        threshold = self.delta * abs(cached)
        if gain <= 0 or gain < threshold:
            return _Candidate(np.inf, interval, None, cached)  # mark dead
        return _Candidate(-gain, interval, children, cached)

    # ------------------------------------------------------------------
    # Local 'searches'
    # ------------------------------------------------------------------

    def _local_best_fs(self):
        heap: list[_Candidate] = []
        for iv in self._get_initial_intervals():
            heapq.heappush(heap, self._evaluate(iv))

        iteration = 0
        while heap and iteration < self.iterations:
            if sum(1 for c in heap if c.children is not None) == 0:
                break
            if len([c for c in heap if True]) >= self.max_intervals:
                break

            top = heapq.heappop(heap)
            if top.children is None:  # dead
                # count for |cover|, but don't re-split
                heapq.heappush(heap, top)
                continue

            c0, c1 = top.children
            heapq.heappush(heap, self._evaluate(c0))
            heapq.heappush(heap, self._evaluate(c1))
            iteration += 1
        return [c.interval for c in heap]

    def _local_bfs(self):
        queue: deque[ICInterval] = deque(self._get_initial_intervals())
        terminal: list[ICInterval] = []

        iteration = 0
        while queue and iteration < self.iterations:
            if len(queue) + len(terminal) >= self.max_intervals:
                break

            iv = queue.popleft()
            cand = self._evaluate(iv)
            if cand.children is None:
                terminal.append(iv)
                continue

            c0, c1 = cand.children
            # add to end
            queue.append(c0)
            queue.append(c1)
            iteration += 1
            # TODO: As implemented iterations are basically just max_intervals
            # should instead make iterations the number of times we parse through each level
            # of the splitting tree -- still useless for DFS?

        terminal.extend(queue)
        return terminal

    def _local_dfs(self):
        stack: list[ICInterval] = list(self._get_initial_intervals())
        terminal: list[ICInterval] = []

        iteration = 0
        while stack and iteration < self.iterations:
            if len(stack) + len(terminal) >= self.max_intervals:
                break

            iv = stack.pop()
            cand = self._evaluate(iv)
            if cand.children is None:
                terminal.append(iv)
                continue

            c_left, c_right = cand.children
            # left child is explored next (DFS order)
            stack.append(c_right)
            stack.append(c_left)
            iteration += 1

        terminal.extend(stack)
        return terminal

    def _local_randomized(self):
        """Local randomized splitting: pick an interval with probability
        proportional to its length, then check a split."""
        active: list[ICInterval] = list(self._get_initial_intervals())
        terminal: list[ICInterval] = []

        iteration = 0
        while active and iteration < self.iterations:
            if len(active) + len(terminal) >= self.max_intervals:
                break

            lengths = np.fromiter(
                (iv.length for iv in active), dtype=float, count=len(active)
            )
            total = lengths.sum()
            if not np.isfinite(total) or total <= 0:
                break
            probs = lengths / total
            idx = int(self.rng.choice(len(active), p=probs))
            iv = active.pop(idx)

            cand = self._evaluate(iv)
            if cand.children is None:
                terminal.append(iv)
                continue

            c0, c1 = cand.children
            active.append(c0)
            active.append(c1)
            iteration += 1

        terminal.extend(active)
        return terminal

    # ------------------------------------------------------------------
    # Global searches
    # ------------------------------------------------------------------

    def _global_best_fs(self):
        cover: list[ICInterval] = self._get_initial_intervals()

        # caches keyed by id(interval);
        #   cleared on merged split
        gain_cache: dict[int, float] = {}
        children_cache: dict[int, tuple[ICInterval, ICInterval]] = {}
        frozen: set[int] = set()  # unsplittable intervals

        iteration = 0
        while iteration < self.iterations:
            if len(cover) >= self.max_intervals:
                break

            # base global score must be recomputed every iteration:
            bic_before = self._score(cover)
            if not np.isfinite(bic_before):
                break

            best_idx = -1
            best_gain = 0.0  # require strictly positive gain to split
            # This is different from the paper's implementation but
            # splitting in these cases can result in a disconnected graph.
            # and morally I am opposed to splitting for no reason.
            best_children: tuple[ICInterval, ICInterval] | None = None

            for idx, iv in enumerate(cover):
                key = id(iv)
                if key in frozen:
                    continue

                if key in gain_cache:
                    gain = gain_cache[key]
                    children = children_cache[key]
                else:
                    children = self._split(iv)
                    trial = cover[:idx] + list(children) + cover[idx + 1 :]
                    bic_after = self._score(trial)
                    if not np.isfinite(bic_after):
                        frozen.add(key)
                        continue
                    gain = bic_after - bic_before
                    threshold = self.delta * abs(bic_before)
                    if gain <= 0 or gain < threshold:
                        # gain >= threshold then we could split
                        frozen.add(key)
                        continue
                    gain_cache[key] = gain
                    children_cache[key] = children

                if gain > best_gain:
                    # require actual gain
                    best_gain = gain
                    best_idx = idx
                    best_children = children

            if best_idx < 0 or best_children is None:
                break  # no split improves the score enough

            parent = cover[best_idx]
            cover = cover[:best_idx] + list(best_children) + cover[best_idx + 1 :]

            # invalidate old cache entries
            pa, pb = parent.lower_bound, parent.upper_bound
            for iv in cover:
                if not (iv.upper_bound <= pa and iv.lower_bound >= pb):
                    k = id(iv)
                    gain_cache.pop(k, None)
                    children_cache.pop(k, None)
                    frozen.discard(k)

            iteration += 1

        return cover

    def _try_global_split(
        self,
        cover: list[ICInterval],
        idx: int,
        bic_before: float,
    ) -> tuple[ICInterval, ICInterval] | None:
        """Attempt to split cover[idx]; return children if it improves the
        global score, else None."""
        iv = cover[idx]
        children = self._split(iv)
        trial = cover[:idx] + list(children) + cover[idx + 1 :]
        bic_after = self._score(trial)
        if not np.isfinite(bic_after):
            return None
        gain = bic_after - bic_before
        threshold = self.delta * abs(bic_before)
        if gain <= 0 or gain < threshold:
            return None
        return children

    def _global_bfs(self):
        cover: list[ICInterval] = self._get_initial_intervals()
        queue: deque[ICInterval] = deque(cover)

        iteration = 0
        while queue and iteration < self.iterations:
            if len(cover) >= self.max_intervals:
                break

            iv = queue.popleft()
            idx = next((i for i, x in enumerate(cover) if x is iv), -1)
            if idx < 0:
                continue  # interval no longer in cover (shouldn't happen)

            bic_before = self._score(cover)
            if not np.isfinite(bic_before):
                break

            result = self._try_global_split(cover, idx, bic_before)
            if result is None:
                continue  # leave iv as a terminal in the cover

            cover = cover[:idx] + list(result) + cover[idx + 1 :]
            queue.append(result[0])
            queue.append(result[1])
            iteration += 1

        return cover

    def _global_dfs(self):
        cover: list[ICInterval] = self._get_initial_intervals()
        stack: list[ICInterval] = list(cover)

        iteration = 0
        while stack and iteration < self.iterations:
            if len(cover) >= self.max_intervals:
                break

            iv = stack.pop()
            idx = next((i for i, x in enumerate(cover) if x is iv), -1)
            if idx < 0:
                continue

            bic_before = self._score(cover)
            if not np.isfinite(bic_before):
                break

            result = self._try_global_split(cover, idx, bic_before)
            if result is None:
                continue

            cover = cover[:idx] + list(result) + cover[idx + 1 :]
            # push so left child is explored next
            stack.append(result[1])
            stack.append(result[0])
            iteration += 1

        return cover

    def _global_randomized(self):
        """Global randomized splitting: select an interval with probability
        proportional to its length, then attempt a global-score split. Once
        an interval is found not to improve the score it is frozen out of
        further sampling."""
        cover: list[ICInterval] = self._get_initial_intervals()
        frozen: set[int] = set()

        iteration = 0
        while iteration < self.iterations:
            if len(cover) >= self.max_intervals:
                break

            candidates = [(i, iv) for i, iv in enumerate(cover) if id(iv) not in frozen]
            if not candidates:
                break

            lengths = np.fromiter(
                (iv.length for _, iv in candidates),
                dtype=float,
                count=len(candidates),
            )
            total = lengths.sum()
            if not np.isfinite(total) or total <= 0:
                break
            probs = lengths / total
            pick = int(self.rng.choice(len(candidates), p=probs))
            idx, iv = candidates[pick]

            bic_before = self._score(cover)
            if not np.isfinite(bic_before):
                break

            result = self._try_global_split(cover, idx, bic_before)
            if result is None:
                frozen.add(id(iv))
                continue

            cover = cover[:idx] + list(result) + cover[idx + 1 :]

            # unfreeze intervals not inside parent
            pa, pb = iv.lower_bound, iv.upper_bound
            for x in cover:
                if not (x.upper_bound <= pa and x.lower_bound >= pb):
                    frozen.discard(id(x))

            iteration += 1
            # again, how is this different from max_intervals?

        return cover
