"""Rare event simulator utilities.

This module provides classes and functions to sample rare error
configurations and compute logical error rates.
"""

from functools import reduce
import pickle
import os
import random
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pymatching
from stim import DetectorErrorModel

from utils import get_decode_info_from_circuit

class ErrorChain:
    """Representation of a set of error sources and the resulting syndrome.

    Attributes:
        det_mat: Detector matrix (numpy array) indexed by error source.
        error_sources: Set of integer indices for active error sources.
        syndrome: Boolean numpy array of detector outcomes produced by errors.
    """

    def __init__(
        self,
        det_mat: np.ndarray,
        initial_error_sources: Optional[Iterable[int]] = None,
    ) -> None:
        """Initialize the ErrorChain.

        Args:
            det_mat: Detector matrix.
            initial_error_sources: Optional iterable of initial error source indices.
        """
        self.det_mat: np.ndarray = det_mat
        self.error_sources: Set[int] = set(initial_error_sources) if initial_error_sources else set()
        self.syndrome: np.ndarray = self.calculate_syndrome()

    def calculate_syndrome(self) -> np.ndarray:
        """Compute the syndrome (detector outcomes) for the current error set.

        Returns:
            A boolean numpy array of detector outcomes.
        """
        syndrome = np.zeros(self.det_mat.shape[1], dtype=bool)
        for e_ix in self.error_sources:
            syndrome ^= self.det_mat[e_ix, :]
        return syndrome

    def add_error_source(self, e_ix: Any) -> None:
        """Toggle one or multiple error sources and update the syndrome.

        Args:
            e_ix: Single index (int) or iterable of indices to toggle.
        """
        if isinstance(e_ix, (list, set, tuple)):
            for e in e_ix:
                if e in self.error_sources:
                    self.error_sources.remove(e)
                else:
                    self.error_sources.add(e)
                self.syndrome ^= self.det_mat[e, :]
        else:
            if e_ix in self.error_sources:
                self.error_sources.remove(e_ix)
            else:
                self.error_sources.add(e_ix)
            self.syndrome ^= self.det_mat[e_ix, :]

    def weight(self) -> int:
        """Return the weight (number of active error sources)."""
        return len(self.error_sources)

    def copy(self) -> "ErrorChain":
        """Return a shallow copy of this ErrorChain (shares det_mat)."""
        new_chain = ErrorChain(self.det_mat, set(self.error_sources))
        new_chain.syndrome = self.syndrome.copy()
        return new_chain

    def __contains__(self, e: int) -> bool:
        """Return True if error source index e is active."""
        return e in self.error_sources

def process_chunk(args: Tuple[Any, ...]) -> Dict[Tuple[int, ...], int]:
    """Process a chunk of seed configurations for multiprocessing.

    Args:
        args: Tuple containing (seed_chunk, det_mat, obs_mat, shots, burn_in, w, dem).

    Returns:
        Mapping from error-index-tuples to observed counts.
    """
    seed_chunk, det_mat, obs_mat, shots, burn_in, w, dem = args
    local_samples: Dict[Tuple[int, ...], int] = {}
    rng = np.random.default_rng()

    for error in seed_chunk:
        E_in = ErrorChain(det_mat, error)
        flip_in = reduce(lambda x, y: x ^ y, [obs_mat[e_ix] for e_ix in error])
        for i in range(shots):
            if i >= burn_in:
                E_in, flip_in, _ = metropolis_one_step(
                    E_in, flip_in, w, dem, rng, det_mat, obs_mat
                )
                err_ixs = tuple(E_in.error_sources)
                local_samples[err_ixs] = local_samples.get(err_ixs, 0) + 1

    return local_samples

def metropolis_one_step(
    E_in: ErrorChain,
    flip_in: np.ndarray,
    w: np.ndarray,
    dem: DetectorErrorModel,
    rng: np.random.Generator,
    det_mat: np.ndarray,
    obs_mat: np.ndarray,
) -> Tuple[ErrorChain, np.ndarray, bool]:
    """Perform one Metropolis step (top-level helper used without RareEventSimulator).
    """
    e_ix = rng.integers(det_mat.shape[0])
    E_trial = E_in.copy()
    E_trial.add_error_source(e_ix)
    flip_trial = flip_in ^ obs_mat[e_ix]
    pi_ratio = (1 - w[e_ix]) / w[e_ix] if e_ix in E_in else w[e_ix] / (1 - w[e_ix])
    q = min(1, pi_ratio)
    if rng.random() < q and is_fail(E_trial, flip_trial, dem, det_mat):
        return E_trial, flip_trial, True
    return E_in, flip_in, False

def is_fail(
    error_chain: ErrorChain, actual_flip: np.ndarray, dem: DetectorErrorModel, det_mat: np.ndarray
) -> bool:
    """Return True if decoding the syndrome does not match the actual flip.

    Args:
        error_chain: ErrorChain instance.
        actual_flip: Observed logical flip (boolean array or scalar boolean).
        dem: DetectorErrorModel used to build the matching graph.
        det_mat: Detector matrix (not used here but kept for API compatibility).
    """
    matcher = pymatching.Matching.from_detector_error_model(dem)
    prediction = matcher.decode(error_chain.syndrome)
    return actual_flip != prediction

def process_Rj(args: Tuple[Any, ...]) -> float:
    """Compute R_j for one pair of adjacent probability partitions.

    Args:
        args: Tuple containing (w1, w2, E1, E2, C_vals, prod_no_err_w1, prod_no_err_w2).

    Returns:
        Chosen C value (float) that balances two estimators.
    """
    w1, w2, E1, E2, C_vals, prod_no_err_w1, prod_no_err_w2 = args
    sum_E1, sum_E2 = sum(E1.values()), sum(E2.values())
    y1: List[float] = []
    y2: List[float] = []

    Aj1 = {
        error: RareEventSimulator.Aj_static(w1, w2, error) * (prod_no_err_w2 / prod_no_err_w1)
        for error in E1
    }
    Aj2 = {
        error: RareEventSimulator.Aj_static(w1, w2, error) * (prod_no_err_w2 / prod_no_err_w1)
        for error in E2
    }

    Aj1_array = np.array([Aj1[error] for error in E1])
    Aj2_array = np.array([Aj2[error] for error in E2])
    n1_array = np.array([n for n in E1.values()])
    n2_array = np.array([n for n in E2.values()])

    for C in C_vals:
        val1 = np.sum(n1_array * RareEventSimulator.g_static(C * Aj1_array)) / sum_E1
        val2 = np.sum(n2_array * RareEventSimulator.g_static(1 / (C * Aj2_array))) / sum_E2
        y1.append(val1)
        y2.append(val2)

    indx = np.argmin(np.abs(np.array(y1) - np.array(y2)))
    return float(C_vals[indx])

class RareEventSimulator():
    def __init__(self, d, circuit_p):
        self.d = d
        self.circuit_p = circuit_p
        self.det_mat, self.obs_mat, _, _ = get_decode_info_from_circuit(circuit_p(1e-3), decompose_errors=False)
        self.num_err_sources = self.det_mat.shape[0]
        self.num_detectors = self.det_mat.shape[1]
    
    def is_fail(self, error_chain: ErrorChain, actual_flip: bool, dem: DetectorErrorModel):
        matcher = pymatching.Matching.from_detector_error_model(dem)
        prediction = matcher.decode(error_chain.syndrome)
        return actual_flip != prediction

    def metropolis_one_step(
        self,
        E_in: ErrorChain,
        flip_in: np.ndarray,
        w: np.ndarray,
        dem: DetectorErrorModel,
        rng: np.random.Generator,
    ) -> Tuple[ErrorChain, np.ndarray, bool]:
        """Perform one Metropolis step using this simulator's detector/obs matrices.

        Returns a tuple: (accepted_error_chain, flip, was_accepted).
        """
        e_ix = rng.integers(self.num_err_sources)
        E_trial = E_in.copy()
        E_trial.add_error_source(e_ix)
        flip_trial = flip_in ^ self.obs_mat[e_ix]
        pi_ratio = (1 - w[e_ix]) / w[e_ix] if e_ix in E_in else w[e_ix] / (1 - w[e_ix])
        q = min(1, pi_ratio)
        if rng.random() < q and self.is_fail(E_trial, flip_trial, dem):
            return E_trial, flip_trial, True
        return E_in, flip_in, False

    def generate_seed_configs(self, num, dem):
        err_configs = []
        # Add the logical operator to the initial uncorrectable chains.
        err_configs.append([self.num_err_sources + e_ix for e_ix in range(-1,-self.d-1,-1)])

        sampled_configs = 1
        while sampled_configs < num:
            err_sources = random.sample(range(self.num_err_sources), random.randint(1, self.d))
            E = ErrorChain(self.det_mat, err_sources)
            flip = reduce(lambda x, y: x ^ y, [self.obs_mat[e_ix] for e_ix in err_sources])
            if self.is_fail(E, flip, dem):
                err_configs.append(err_sources)
                sampled_configs += 1
        return err_configs

    def sample(self, p, num_seed, shots, burn_in, use_multiprocessing=False):
        """
        Samples error configurations at noise rate p.

        :return: Error configurations in the form {(e1_ix, ..., ek_ix): num}.
        :rtype: dict
        """
        # Get dem and weights of matching graph.
        circuit = self.circuit_p(p)
        w = get_decode_info_from_circuit(circuit, decompose_errors=False)[2]
        dem = circuit.detector_error_model(decompose_errors=True)

        # Generate seed configurations.
        seed_configs = self.generate_seed_configs(num_seed, dem)

        if use_multiprocessing:
            # Run metropolis with multiprocessing.
            num_cpus = cpu_count()
            chunk_size = max(1, len(seed_configs) // num_cpus)

            with Pool(num_cpus) as pool:
                chunks = [seed_configs[i:i + chunk_size] for i in range(0, len(seed_configs), chunk_size)]
                results = pool.map(process_chunk, [(chunk, self.det_mat, self.obs_mat, shots, burn_in, w, dem) for chunk in chunks])

            # Combine results from all processes.
            samples = {}
            for result in results:
                for err_ixs, count in result.items():
                    if err_ixs not in samples:
                        samples[err_ixs] = count
                    else:
                        samples[err_ixs] += count
        else:
            # Run metropolis without multiprocessing.
            samples = {}
            rng = np.random.default_rng()
            for error in seed_configs:
                E_in = ErrorChain(self.det_mat, error)
                flip_in = reduce(lambda x, y: x ^ y, [self.obs_mat[e_ix] for e_ix in error])
                for i in range(shots):
                    if i >= burn_in:
                        E_in, flip_in, is_new = metropolis_one_step(E_in, flip_in, w, dem, rng, self.det_mat, self.obs_mat)
                        err_ixs = tuple(E_in.error_sources)
                        if err_ixs not in samples:
                            samples[err_ixs] = 1
                        else:
                            samples[err_ixs] += 1

        return samples

    def generate_probs(self, p_target, p_known):
        """
        Generates the array of physical error probabilities, p_t < ...< p2 < p1.
        wj = sum_{e \in edges} pj(e) is the sum of all probabilities in the DEM.
        """
        probs = []
        ws = []
        p_current = p_target
        probs.append(p_current)
        while True:
            w = get_decode_info_from_circuit(self.circuit_p(p_current))[2]
            ws.append(w)
            wj = max(self.d/2, np.sum(w))
            p_next = p_current * (2 ** (1 / np.sqrt(wj)))
            if p_next > p_known:
                probs[-1] = p_known
                ws[-1] = get_decode_info_from_circuit(self.circuit_p(p_known))[2]
                break
            probs.append(p_next)
            p_current = p_next
        return probs, ws
    
    def prob_err(self, error, w):
        prob = 1
        for e_ix in range(len(w)):
            if e_ix in error:
                prob *= w[e_ix]
            else:
                prob *= 1 - w[e_ix]
        return prob

    def Aj(self, w1, w2, error):
        """
        P(error)_w2 / P(error)_w1
        NEED: p1 < p2
        """
        prob = 1
        for e_ix in error:
            prob *= (w2[e_ix] * (1-w1[e_ix]) / (w1[e_ix] * (1-w2[e_ix])))
        return prob
        # return self.prob_err(error, w2) / self.prob_err(error, w1)

    @staticmethod
    def Aj_static(w1, w2, error):
        prob = 1
        for e_ix in error:
            prob *= (w2[e_ix] / (1-w2[e_ix])) * ((1-w1[e_ix]) / w1[e_ix])
        return prob

    def g(self, x):
        return 1 / (1 + x)

    @staticmethod
    def g_static(x):
        return 1 / (1 + x)
    
    def save_samples(self, p_target, p_known, num_seed, shots, burn_in, use_multiprocessing=False):
        """
        Creates range of p values, from p_to_find to p_known, and samples error
        configurations at each point.
        """
        probs, ws = self.generate_probs(p_target, p_known)
        # Ensure directories exist and save weights
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./samples", exist_ok=True)
        with open(f"./weights/{self.d}_{p_target}.pkl", "wb") as f:
            pickle.dump(ws, f)
        for ix, p in enumerate(probs):
            samples = self.sample(p, num_seed, shots, burn_in, use_multiprocessing)
            with open(f'./samples/{self.d}_{p_target}_{ix}.pkl', 'wb') as f:
                pickle.dump(samples, f)

    def save_single_sample(self, p_target, num_seed, shots, p, ix, burn_in, use_multiprocessing=False):
        """
        Creates range of p values, from p_to_find to p_known, and samples error
        configurations at each point.
        """
        samples = self.sample(p, num_seed, shots, burn_in, use_multiprocessing)
        os.makedirs("./samples", exist_ok=True)
        with open(f'./samples/{self.d}_{p_target}_{ix}.pkl', 'wb') as f:
            pickle.dump(samples, f)

    def prod_no_err(self, w):
        """
        Returns the probability of no error occurring.
        """
        prob = 1
        for e_ix in range(len(w)):
            prob *= (1 - w[e_ix])
        return prob

    def LER(self, p_target, p_known, LER_known, use_multiprocessing=False):
        """
        Reads saved error configurations and uses the splitting method to calculate
        the logical error rate at p_target.
        """
        # Generate partition of physical error rates.
        # probs, ws = self.generate_probs(p_target, p_known)
        with open(f"./weights/{self.d}_{p_target}.pkl", "rb") as f:
            ws = pickle.load(f)
        prod_no_err_ws = [self.prod_no_err(w) for w in ws]

        # Load in saved error configurations.
        sampled_E = []  # List of dictionaries.
        for ix in range(len(ws)):
            with open(f'./samples/{self.d}_{p_target}_{ix}.pkl', 'rb') as f:
                sampled_E.append(pickle.load(f))

        # Calculate LER at p_target.
        P = LER_known
        C_vals = np.linspace(0.0001, 1, 100)

        if use_multiprocessing:
            num_cpus = cpu_count()
            with Pool(num_cpus) as pool:
                args = [(ws[j], ws[j + 1], sampled_E[j + 1], sampled_E[j], C_vals, prod_no_err_ws[j], prod_no_err_ws[j + 1]) for j in range(len(ws) - 1)]
                Rjs = pool.map(process_Rj, args)
            for Rj in Rjs:
                P *= Rj
        else:
            for j in range(len(ws) - 1):
                w1, w2 = ws[j], ws[j + 1]
                prod_no_err_w1, prod_no_err_w2 = prod_no_err_ws[j], prod_no_err_ws[j + 1]
                E1, E2 = sampled_E[j + 1], sampled_E[j]
                sum_E1, sum_E2 = sum(E1.values()), sum(E2.values())
                y1, y2 = [], []

                Aj1 = {error: self.Aj(w1, w2, error) * (prod_no_err_w2 / prod_no_err_w1) for error in E1}
                Aj2 = {error: self.Aj(w1, w2, error) * (prod_no_err_w2 / prod_no_err_w1) for error in E2}

                Aj1_array = np.array([Aj1[error] for error in E1])
                Aj2_array = np.array([Aj2[error] for error in E2])
                n1_array = np.array([n for n in E1.values()])
                n2_array = np.array([n for n in E2.values()])

                for C in C_vals:
                    val1 = np.sum(n1_array * self.g(C * Aj1_array)) / sum_E1
                    val2 = np.sum(n2_array * self.g(1 / (C * Aj2_array))) / sum_E2
                    y1.append(val1)
                    y2.append(val2)

                indx = np.argmin(np.abs(np.array(y1) - np.array(y2)))
                C_star = C_vals[indx]  # This is R_j
                P *= C_star

        return P

