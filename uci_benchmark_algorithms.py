from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


EPS = np.finfo(np.float64).eps
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 5000
DEFAULT_TOL = 1e-5


def standardize(X):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def pairwise_squared_distances(X):
    X = np.asarray(X, dtype=np.float64)
    squared_norms = np.sum(X * X, axis=1)
    d2 = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def pairwise_distances(X):
    return np.sqrt(pairwise_squared_distances(X))


def sigmoid(z):
    z = np.clip(z, -700, 700)
    return 1.0 / (1.0 + np.exp(-z))


def safe_row_normalize(A):
    A = np.asarray(A, dtype=np.float64)
    row_sum = A.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= EPS
    out = A / np.maximum(row_sum, EPS)
    if np.any(bad):
        out[bad] = 1.0 / A.shape[1]
    return out


def deterministic_label_codes(values):
    categories = sorted(pd.unique(values))
    return pd.Categorical(values, categories=categories, ordered=True).codes


def update_fcm_memberships_from_d2(d2, m):
    d2 = np.asarray(d2, dtype=np.float64)
    n_samples, n_clusters = d2.shape
    U = np.zeros_like(d2)

    zero_mask = d2 <= EPS
    rows_with_zero = np.any(zero_mask, axis=1)
    if np.any(rows_with_zero):
        counts = np.sum(zero_mask[rows_with_zero], axis=1, keepdims=True)
        U[rows_with_zero] = zero_mask[rows_with_zero] / counts

    nonzero_rows = ~rows_with_zero
    if np.any(nonzero_rows):
        temp = np.maximum(d2[nonzero_rows], EPS) ** (-1.0 / (m - 1.0))
        U[nonzero_rows] = temp / np.sum(temp, axis=1, keepdims=True)

    return U


def distances_to_centers_squared(X, centers):
    return np.column_stack([np.sum((X - center) ** 2, axis=1) for center in centers])


def fcm_centers(X, U, m):
    weights = U ** m
    denominator = np.maximum(weights.sum(axis=0)[:, None], EPS)
    return (weights.T @ X) / denominator


def fcm_objective(X, U, centers, m):
    d2 = distances_to_centers_squared(X, centers)
    return float(np.sum((U ** m) * d2))


def partition_coefficient(U):
    return float(np.sum(U ** 2) / U.shape[0])


def partition_entropy(U):
    return float(-np.sum(U * np.log(np.maximum(U, EPS))) / U.shape[0])


def contingency_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_values, true_inv = np.unique(y_true, return_inverse=True)
    pred_values, pred_inv = np.unique(y_pred, return_inverse=True)
    table = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    np.add.at(table, (true_inv, pred_inv), 1)
    return table


def purity_score(y_true, y_pred):
    table = contingency_matrix(y_true, y_pred)
    return float(table.max(axis=0).sum() / table.sum())


def normalized_mutual_info(y_true, y_pred):
    table = contingency_matrix(y_true, y_pred).astype(np.float64)
    n = table.sum()
    if n == 0:
        return 0.0

    p = table / n
    p_true = p.sum(axis=1, keepdims=True)
    p_pred = p.sum(axis=0, keepdims=True)

    expected = p_true @ p_pred
    mask = p > 0
    mi = float(np.sum(p[mask] * np.log(p[mask] / expected[mask])))

    h_true = float(-np.sum(p_true[p_true > 0] * np.log(p_true[p_true > 0])))
    h_pred = float(-np.sum(p_pred[p_pred > 0] * np.log(p_pred[p_pred > 0])))
    denom = h_true + h_pred
    if denom <= EPS:
        return 1.0
    return float(2.0 * mi / denom)


def comb2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def adjusted_rand_index(y_true, y_pred):
    table = contingency_matrix(y_true, y_pred).astype(np.float64)
    n = table.sum()
    if n < 2:
        return 1.0

    sum_comb = comb2(table).sum()
    row_comb = comb2(table.sum(axis=1)).sum()
    col_comb = comb2(table.sum(axis=0)).sum()
    total_comb = comb2(n)
    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if abs(denom) <= EPS:
        return 0.0
    return float((sum_comb - expected) / denom)


def silhouette_score_np(X, labels):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size <= 1 or unique.size >= X.shape[0]:
        return 0.0

    distances = pairwise_distances(X)
    scores = np.zeros(X.shape[0], dtype=np.float64)

    for i in range(X.shape[0]):
        same = labels == labels[i]
        same[i] = False
        if np.any(same):
            a = float(np.mean(distances[i, same]))
        else:
            a = 0.0

        b = np.inf
        for label in unique:
            if label == labels[i]:
                continue
            other = labels == label
            if np.any(other):
                b = min(b, float(np.mean(distances[i, other])))

        if not np.isfinite(b):
            scores[i] = 0.0
        else:
            denom = max(a, b)
            scores[i] = 0.0 if denom <= EPS else (b - a) / denom

    return float(np.mean(scores))


def evaluate_memberships(X, y_true, U):
    U = safe_row_normalize(U)
    labels = np.argmax(U, axis=1)
    return {
        "pc": partition_coefficient(U),
        "pe": partition_entropy(U),
        "sil": silhouette_score_np(X, labels),
        "pur": purity_score(y_true, labels),
        "nmi": normalized_mutual_info(y_true, labels),
        "ari": adjusted_rand_index(y_true, labels),
    }


@dataclass
class FitResult:
    method: str
    converged: bool
    iterations: int
    total_time_s: float
    objective: float
    U: np.ndarray
    centers: np.ndarray
    extra: dict


class FuzzyCMeans:
    def __init__(self, n_clusters, m=2.0, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE, init_centers=None):
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.init_centers = init_centers

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        if self.init_centers is None:
            rng = np.random.default_rng(self.random_state)
            U = rng.random((n_samples, self.n_clusters))
            U = safe_row_normalize(U)
        else:
            centers = np.asarray(self.init_centers, dtype=np.float64)
            U = update_fcm_memberships_from_d2(distances_to_centers_squared(X, centers), self.m)

        converged = False
        centers = None
        for iteration in range(1, self.max_iter + 1):
            U_old = U.copy()
            centers = fcm_centers(X, U, self.m)
            d2 = distances_to_centers_squared(X, centers)
            U = update_fcm_memberships_from_d2(d2, self.m)
            if np.linalg.norm(U - U_old, ord="fro") < self.tol:
                converged = True
                break

        objective = fcm_objective(X, U, centers, self.m)
        return FitResult(
            method="FCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=U,
            centers=centers,
            extra={},
        )


class PossibilisticCMeans:
    def __init__(self, n_clusters, m=2.0, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE):
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    @staticmethod
    def _eta_from_fcm(X, U, centers, m):
        d2 = distances_to_centers_squared(X, centers)
        weights = U ** m
        return np.maximum(np.sum(weights * d2, axis=0) / np.maximum(weights.sum(axis=0), EPS), EPS)

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)

        fcm = FuzzyCMeans(
            self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        ).fit(X)
        centers = fcm.centers.copy()
        eta = self._eta_from_fcm(X, fcm.U, centers, self.m)
        d2 = distances_to_centers_squared(X, centers)
        T = 1.0 / (1.0 + (d2 / eta[None, :]) ** (1.0 / (self.m - 1.0)))

        converged = False
        for iteration in range(1, self.max_iter + 1):
            T_old = T.copy()
            weights = T ** self.m
            centers = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], EPS)
            d2 = distances_to_centers_squared(X, centers)
            T = 1.0 / (1.0 + (d2 / eta[None, :]) ** (1.0 / (self.m - 1.0)))
            if np.linalg.norm(T - T_old, ord="fro") < self.tol:
                converged = True
                break

        U_eval = safe_row_normalize(T)
        objective = float(np.sum((T ** self.m) * d2) + np.sum(eta[None, :] * ((1.0 - T) ** self.m)))
        return FitResult(
            method="PCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=U_eval,
            centers=centers,
            extra={"eta_mean": float(np.mean(eta))},
        )


class PossibilisticFuzzyCMeans:
    def __init__(self, n_clusters, m=2.0, nu=2.0, a=1.0, b=1.0, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE):
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.nu = float(nu)
        self.a = float(a)
        self.b = float(b)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    @staticmethod
    def _eta(X, U, centers, m):
        d2 = distances_to_centers_squared(X, centers)
        weights = U ** m
        return np.maximum(np.sum(weights * d2, axis=0) / np.maximum(weights.sum(axis=0), EPS), EPS)

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)

        fcm = FuzzyCMeans(
            self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        ).fit(X)
        U = fcm.U.copy()
        centers = fcm.centers.copy()
        eta = self._eta(X, U, centers, self.m)
        d2 = distances_to_centers_squared(X, centers)
        T = 1.0 / (1.0 + ((self.b * d2) / eta[None, :]) ** (1.0 / (self.nu - 1.0)))

        converged = False
        for iteration in range(1, self.max_iter + 1):
            U_old = U.copy()
            T_old = T.copy()

            weights = self.a * (U ** self.m) + self.b * (T ** self.nu)
            centers = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], EPS)
            d2 = distances_to_centers_squared(X, centers)

            U = update_fcm_memberships_from_d2(d2, self.m)
            eta = self._eta(X, U, centers, self.m)
            T = 1.0 / (1.0 + ((self.b * d2) / eta[None, :]) ** (1.0 / (self.nu - 1.0)))

            delta = max(np.linalg.norm(U - U_old, ord="fro"), np.linalg.norm(T - T_old, ord="fro"))
            if delta < self.tol:
                converged = True
                break

        objective = float(
            self.a * np.sum((U ** self.m) * d2)
            + self.b * np.sum((T ** self.nu) * d2)
            + np.sum(eta[None, :] * ((1.0 - T) ** self.nu))
        )
        return FitResult(
            method="PFCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=U,
            centers=centers,
            extra={"eta_mean": float(np.mean(eta))},
        )


def density_peak_initial_centers(X, n_clusters):
    X = np.asarray(X, dtype=np.float64)
    d2 = pairwise_squared_distances(X)
    d = np.sqrt(d2)
    upper = d[np.triu_indices(X.shape[0], k=1)]
    positive = upper[upper > 0]
    dc = np.percentile(positive, 2) if positive.size else EPS
    dc2 = max(dc * dc, EPS)

    rho = np.exp(-d2 / dc2).sum(axis=1) - 1.0
    rho = np.maximum(rho, 0.0)
    order = np.argsort(-rho)
    delta = np.zeros(X.shape[0], dtype=np.float64)

    max_distance = np.max(d) if d.size else 0.0
    delta[order[0]] = max_distance
    for position in range(1, len(order)):
        idx = order[position]
        higher = order[:position]
        delta[idx] = np.min(d[idx, higher])

    score = rho * delta
    center_indices = np.argsort(-score)[:n_clusters]
    return X[center_indices], {"dpc_dc": float(dc), "dpc_score_mean": float(np.mean(score))}


class DPCFuzzyCMeans:
    def __init__(self, n_clusters, m=2.0, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL):
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X):
        start = perf_counter()
        centers, extra = density_peak_initial_centers(X, self.n_clusters)
        fcm = FuzzyCMeans(
            self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=0,
            init_centers=centers,
        ).fit(X)
        fcm.method = "DPC-FCM"
        fcm.total_time_s = perf_counter() - start
        fcm.extra.update(extra)
        return fcm


class EntropyRegularizedFCM:
    def __init__(self, n_clusters, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE, lam=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.lam = lam

    @staticmethod
    def _soft_memberships(d2, lam):
        scores = -d2 / max(lam, EPS)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), EPS)

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        U = rng.random((X.shape[0], self.n_clusters))
        U = safe_row_normalize(U)

        if self.lam is None:
            all_d2 = pairwise_squared_distances(X)
            upper = all_d2[np.triu_indices(X.shape[0], k=1)]
            positive = upper[upper > 0]
            lam = float(np.median(positive) / 2.0) if positive.size else 1.0
        else:
            lam = float(self.lam)

        converged = False
        centers = None
        for iteration in range(1, self.max_iter + 1):
            U_old = U.copy()
            denominator = np.maximum(U.sum(axis=0)[:, None], EPS)
            centers = (U.T @ X) / denominator
            d2 = distances_to_centers_squared(X, centers)
            U = self._soft_memberships(d2, lam)
            if np.linalg.norm(U - U_old, ord="fro") < self.tol:
                converged = True
                break

        objective = float(np.sum(U * distances_to_centers_squared(X, centers)) + lam * np.sum(U * np.log(np.maximum(U, EPS))))
        return FitResult(
            method="eFCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=U,
            centers=centers,
            extra={"lambda": lam},
        )


class DensityAdaptiveFuzzyCMeans:
    def __init__(self, n_clusters, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, m_low=1.2, m_high=4.0, alpha=5.0, random_state=DEFAULT_RANDOM_STATE):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.m_low = float(m_low)
        self.m_high = float(m_high)
        self.alpha = float(alpha)
        self.random_state = random_state

    def _density_adaptive_values(self, X):
        d2 = pairwise_squared_distances(X)
        pairwise = np.sqrt(d2[np.triu_indices(X.shape[0], k=1)])
        positive = pairwise[pairwise > 0]
        dc = np.percentile(positive, 2) if positive.size else EPS
        if dc <= 0:
            dc = np.min(positive)

        kernel = np.exp(-d2 / max(dc * dc, EPS))
        rho = np.maximum(kernel.sum(axis=1) - 1.0, 0.0)
        rho_hat = np.zeros_like(rho) if rho.max() <= 0 else rho / rho.max()
        m_i = self.m_low + (self.m_high - self.m_low) * sigmoid(-self.alpha * rho_hat)
        e_i = 1.0 / (m_i - 1.0)
        return dc, rho, rho_hat, m_i, e_i

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)
        density_start = perf_counter()
        dc, rho, rho_hat, m_i, e_i = self._density_adaptive_values(X)
        density_time_s = perf_counter() - density_start

        rng = np.random.default_rng(self.random_state)
        U = rng.random((X.shape[0], self.n_clusters))
        U = safe_row_normalize(U)

        converged = False
        centers = None
        objective = np.nan
        iteration_start = perf_counter()
        for iteration in range(1, self.max_iter + 1):
            U_old = U.copy()

            weights = U ** m_i[:, None]
            centers = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], EPS)
            d2 = distances_to_centers_squared(X, centers)

            new_U = np.zeros_like(U)
            zero_mask = d2 <= EPS
            rows_with_zero = np.any(zero_mask, axis=1)
            if np.any(rows_with_zero):
                counts = np.sum(zero_mask[rows_with_zero], axis=1, keepdims=True)
                new_U[rows_with_zero] = zero_mask[rows_with_zero] / counts

            nonzero_rows = ~rows_with_zero
            if np.any(nonzero_rows):
                temp = np.maximum(d2[nonzero_rows], EPS) ** (-e_i[nonzero_rows, None])
                new_U[nonzero_rows] = temp / np.sum(temp, axis=1, keepdims=True)
            U = new_U

            objective = float(np.sum((U ** m_i[:, None]) * d2))
            if np.linalg.norm(U - U_old, ord="fro") < self.tol:
                converged = True
                break

        iteration_time_s = perf_counter() - iteration_start
        return FitResult(
            method="DA-FCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=U,
            centers=centers,
            extra={
                "density_time_s": density_time_s,
                "iteration_time_s": iteration_time_s,
                "dc": float(dc),
                "m_i_min": float(np.min(m_i)),
                "m_i_mean": float(np.mean(m_i)),
                "m_i_max": float(np.max(m_i)),
                "e_i_min": float(np.min(e_i)),
                "e_i_mean": float(np.mean(e_i)),
                "e_i_max": float(np.max(e_i)),
            },
        )


def download_if_missing(data_dir, filename, url):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    path = data_dir / filename
    if not path.exists():
        print(f"Downloading {filename} ...")
        try:
            urlretrieve(url, path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {url}. Manually place the file at {path} and rerun."
            ) from exc
    return path


def load_uci_datasets(data_dir):
    data_dir = Path(data_dir)
    datasets = []

    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    path = download_if_missing(data_dir, "iris.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    df = pd.read_csv(path, names=cols).dropna()
    datasets.append({
        "name": "Iris",
        "X": standardize(df[cols[:-1]].to_numpy()),
        "y": deterministic_label_codes(df["class"]),
        "classes": 3,
        "characteristics": "One separable, two overlapping",
    })

    feature_cols = [f"feature_{i}" for i in range(1, 14)]
    path = download_if_missing(data_dir, "wine.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")
    df = pd.read_csv(path, names=["class"] + feature_cols).dropna()
    datasets.append({
        "name": "Wine",
        "X": standardize(df[feature_cols].to_numpy()),
        "y": deterministic_label_codes(df["class"]),
        "classes": 3,
        "characteristics": "Heterogeneous density, correlated features",
    })

    feature_cols = [f"feature_{i}" for i in range(1, 31)]
    path = download_if_missing(data_dir, "wdbc.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    df = pd.read_csv(path, names=["id", "diagnosis"] + feature_cols).dropna()
    datasets.append({
        "name": "Breast Cancer",
        "X": standardize(df[feature_cols].to_numpy()),
        "y": deterministic_label_codes(df["diagnosis"]),
        "classes": 2,
        "characteristics": "Imbalanced, high-dimensional",
    })

    feature_cols = [f"feature_{i}" for i in range(1, 8)]
    path = download_if_missing(data_dir, "seeds_dataset.txt", "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt")
    df = pd.read_csv(path, sep=r"\s+", names=feature_cols + ["class"], engine="python").dropna()
    datasets.append({
        "name": "Seeds",
        "X": standardize(df[feature_cols].to_numpy()),
        "y": deterministic_label_codes(df["class"]),
        "classes": 3,
        "characteristics": "Balanced, moderate overlap",
    })

    feature_cols = ["ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe"]
    path = download_if_missing(data_dir, "glass.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")
    df = pd.read_csv(path, names=["id"] + feature_cols + ["class"]).dropna()
    datasets.append({
        "name": "Glass",
        "X": standardize(df[feature_cols].to_numpy()),
        "y": deterministic_label_codes(df["class"]),
        "classes": 6,
        "characteristics": "Severe imbalance, high overlap",
    })

    feature_cols = [f"feature_{i}" for i in range(1, 8)]
    path = download_if_missing(data_dir, "ecoli.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data")
    df = pd.read_csv(path, sep=r"\s+", names=["sequence_name"] + feature_cols + ["class"], engine="python").dropna()
    datasets.append({
        "name": "Ecoli",
        "X": standardize(df[feature_cols].to_numpy()),
        "y": deterministic_label_codes(df["class"]),
        "classes": int(df["class"].nunique()),
        "characteristics": "Protein localization, strongly imbalanced multiclass",
    })

    feature_cols = [f"feature_{i}" for i in range(1, 9)]
    path = download_if_missing(data_dir, "yeast.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data")
    df = pd.read_csv(path, sep=r"\s+", names=["sequence_name"] + feature_cols + ["class"], engine="python").dropna()
    datasets.append({
        "name": "Yeast",
        "X": standardize(df[feature_cols].to_numpy()),
        "y": deterministic_label_codes(df["class"]),
        "classes": int(df["class"].nunique()),
        "characteristics": "Protein localization, larger imbalanced multiclass",
    })

    feature_cols = [
        "region_centroid_col",
        "region_centroid_row",
        "region_pixel_count",
        "short_line_density_5",
        "short_line_density_2",
        "vedge_mean",
        "vedge_sd",
        "hedge_mean",
        "hedge_sd",
        "intensity_mean",
        "rawred_mean",
        "rawblue_mean",
        "rawgreen_mean",
        "exred_mean",
        "exblue_mean",
        "exgreen_mean",
        "value_mean",
        "saturation_mean",
        "hue_mean",
    ]
    segment_cols = ["class"] + feature_cols
    segment_train = download_if_missing(
        data_dir,
        "segmentation.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data",
    )
    segment_test = download_if_missing(
        data_dir,
        "segmentation.test",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test",
    )
    segment_frames = [
        pd.read_csv(segment_train, skiprows=5, names=segment_cols),
        pd.read_csv(segment_test, skiprows=5, names=segment_cols),
    ]
    df = pd.concat(segment_frames, ignore_index=True)
    df = df.dropna()
    datasets.append({
        "name": "Image Segmentation",
        "X": standardize(df[feature_cols].to_numpy()),
        "y": deterministic_label_codes(df["class"]),
        "classes": int(df["class"].nunique()),
        "characteristics": "Image-region features, 7 balanced outdoor-scene classes",
    })

    for dataset in datasets:
        dataset["samples"] = int(dataset["X"].shape[0])
        dataset["features"] = int(dataset["X"].shape[1])
        dataset["n_clusters"] = int(dataset["classes"])

    return datasets


def make_model(method, n_clusters, seed, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL):
    if method == "FCM":
        return FuzzyCMeans(n_clusters, m=2.0, max_iter=max_iter, tol=tol, random_state=seed)
    if method == "PCM":
        return PossibilisticCMeans(n_clusters, m=2.0, max_iter=max_iter, tol=tol, random_state=seed)
    if method == "PFCM":
        return PossibilisticFuzzyCMeans(n_clusters, m=2.0, nu=2.0, a=1.0, b=1.0, max_iter=max_iter, tol=tol, random_state=seed)
    if method == "DPC-FCM":
        return DPCFuzzyCMeans(n_clusters, m=2.0, max_iter=max_iter, tol=tol)
    if method == "eFCM":
        return EntropyRegularizedFCM(n_clusters, max_iter=max_iter, tol=tol, random_state=seed)
    if method == "DA-FCM":
        return DensityAdaptiveFuzzyCMeans(n_clusters, max_iter=max_iter, tol=tol, m_low=1.2, m_high=4.0, alpha=5.0, random_state=seed)
    raise ValueError(f"Unknown method: {method}")


def run_single_method(dataset, method, seed, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL):
    model = make_model(method, dataset["n_clusters"], seed, max_iter=max_iter, tol=tol)
    result = model.fit(dataset["X"])
    metrics = evaluate_memberships(dataset["X"], dataset["y"], result.U)

    row = {
        "dataset": dataset["name"],
        "samples": dataset["samples"],
        "features": dataset["features"],
        "classes": dataset["classes"],
        "method": method,
        "seed": seed,
        "converged": result.converged,
        "iterations": result.iterations,
        "total_time_s": result.total_time_s,
        "objective": result.objective,
        **metrics,
    }
    row.update(result.extra)
    return row


def benchmark_all_methods(
    datasets,
    methods=("FCM", "PCM", "PFCM", "DPC-FCM", "eFCM", "DA-FCM"),
    n_init=5,
    random_state=DEFAULT_RANDOM_STATE,
    max_iter=DEFAULT_MAX_ITER,
    tol=DEFAULT_TOL,
):
    all_rows = []
    summary_rows = []

    for dataset in datasets:
        for method in methods:
            run_rows = []
            for init_idx in range(n_init):
                seed = random_state + init_idx
                row = run_single_method(dataset, method, seed, max_iter=max_iter, tol=tol)
                row["init_index"] = init_idx + 1
                run_rows.append(row)
                all_rows.append(row)

            objectives = np.array([row["objective"] for row in run_rows], dtype=np.float64)
            stable_init_order = np.arange(len(run_rows))
            best_idx = int(np.lexsort((stable_init_order, np.nan_to_num(objectives, nan=np.inf)))[0])
            best = run_rows[best_idx].copy()
            best["n_init"] = n_init
            best["best_seed"] = best["seed"]
            summary_rows.append(best)

            print(
                f"{dataset['name']} | {method}: "
                f"PC={best['pc']:.4f}, PE={best['pe']:.4f}, "
                f"Iter={best['iterations']}, Time={best['total_time_s'] * 1000:.1f} ms"
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(all_rows)
