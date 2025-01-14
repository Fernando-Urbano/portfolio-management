import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from itertools import product
from skopt import gp_minimize
from skopt.space import Real, Categorical

from portfolio_management.port_construction import calc_weights, PERIODS_PER_YEAR


# defines the configuration for tunable parameters in the optimization process
PARAM_SPECS = {
    "l1_reg": {"type": Real, "default": 0.0},
    "l2_reg": {"type": Real, "default": 0.0},
    "shrinkage_target": {"type": Categorical, "default": None},
    "shrinkage_factor": {"type": Real, "default": None},
}


# supported evaluation metrics and their optimization direction.
EVAL_METRICS = {
    "mean_return": {"maximize": True},
    "sharpe": {"maximize": True},
    "volatility": {"maximize": False},
}


def _evaluate_performance(
    test_data: pd.DataFrame, weights: pd.Series, metric: str, periods_per_year: int
) -> float:
    """
    Evaluate the test performance given test_data and portfolio weights.

    Parameters
    ----------
    test_data : pd.DataFrame
        Test returns data (rows=time, columns=assets).
    weights : pd.Series
        Portfolio weights for each asset.
    metric : str
        Performance metric to calculate.
    periods_per_year : int
        Number of periods in a year for annualization (e.g. 12 for monthly data).

    Returns
    -------
    float
        Calculated performance value based on the specified performance metric.
    """

    portfolio_returns = test_data @ weights

    if metric == "mean_return":
        mean_return = portfolio_returns.mean()
        if periods_per_year:
            mean_return *= periods_per_year
        return mean_return
    elif metric == "volatility":
        volatility = portfolio_returns.std()
        if periods_per_year:
            volatility *= np.sqrt(periods_per_year)
        return volatility
    elif metric == "sharpe":
        mean_return = portfolio_returns.mean()
        std_dev = portfolio_returns.std()
        if periods_per_year:
            mean_return *= periods_per_year
            std_dev *= np.sqrt(periods_per_year)
        return (mean_return / std_dev) if std_dev != 0 else 0.0

    raise ValueError(
        f"Invalid evaluation metric: '{metric}'. Valid options are: {', '.join(EVAL_METRICS)}."
    )


# CORE TSCV HELPER


def _run_tscv_for_params(
    returns: pd.DataFrame,
    eval_metric: str,
    aggregator: str,
    l1_reg: float,
    l2_reg: float,
    shrinkage_target: Union[str, np.ndarray],
    shrinkage_factor: float,
    window_type: str,
    n_folds: int,
    train_window: int,
    test_window: int,
    step_size: int,
    **calc_weights_kwargs,
) -> float:
    """
    Helper function that runs time-series cross-validation (TSCV) for a single
    hyperparameter combination (e.g., l1_reg, l2_reg, shrinkage_target, shrinkage_factor)
    and returns an aggregated performance score.

    This function slices the data into training and testing windows for each fold,
    calls `calc_weights` to fit portfolio weights on the training set, then evaluates
    out-of-sample performance on the test set. The final score is computed by
    aggregating (mean or median) the per-fold performance values.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns data.
    eval_metric : str
        The evaluation metric (e.g., mean_return, sharpe, volatility). Also used to decide
        whether we maximize or minimize scores.
    aggregator : str
        How to combine the fold-level performance scores. Must be one of ('mean', 'median').
    l1_reg : float
        L1 regularization penalty factor (Lasso). Passed to `calc_weights`.
    l2_reg : float
        L2 regularization penalty factor (Ridge). Passed to `calc_weights`.
    shrinkage_target : Union[str, np.ndarray]
        Covariance shrinkage target. Passed to `calc_weights`.
    shrinkage_factor : float
        Covariance shrinkage intensity. Passed to `calc_weights`.
    window_type : str
        TSCV approach: "rolling" or "expanding".
    n_folds : int
        Number of folds. If None, it is derived automatically from the data length,
        train_window, test_window, and window_type.
    train_window : int
        Number of observations used for the training set.
    test_window : int
        Number of observations used for the testing set.
    step_size : int
        How many observations to move forward after each fold. Defaults to test_window.
    **calc_weights_kwargs
        Additional parameters forwarded to `calc_weights`.

    Returns
    -------
    float
        The aggregated performance score over all TSCV folds. If no valid folds are found,
        returns ±inf depending on whether the metric is maximized or minimized.
    """
    n_rows = len(returns)
    if train_window is None or test_window is None:
        raise ValueError("train_window and test_window must be specified.")
    if step_size is None:
        step_size = test_window

    # automatically calculate n_folds if not specified
    if n_folds is None:
        if window_type == "rolling":
            n_folds = (n_rows - train_window) // step_size
        elif window_type == "expanding":
            n_folds = (n_rows - test_window) // step_size
        else:
            raise ValueError("window_type must be 'rolling' or 'expanding'.")
        # ensure at least one fold
        n_folds = max(n_folds, 1)

    fold_perfs = []
    start_idx = 0

    for fold_i in range(n_folds):
        # define train/test boundaries
        if window_type == "rolling":
            train_start = start_idx
            train_end = train_start + train_window
        elif window_type == "expanding":
            train_start = 0
            train_end = start_idx + train_window
        else:
            raise ValueError("window_type must be 'rolling' or 'expanding'.")

        test_start = train_end
        test_end = test_start + test_window

        # not enough data for another fold
        if test_end > n_rows:
            break

        # slice train/test data
        train_data = returns.iloc[train_start:train_end]
        test_data = returns.iloc[test_start:test_end]

        # fit model
        try:
            weights = calc_weights(
                train_data,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                shrinkage_target=shrinkage_target,
                shrinkage_factor=shrinkage_factor,
                **calc_weights_kwargs,
            )
        except Exception:
            fold_perfs.append(np.nan)
        else:
            # evaluate performance
            perf = _evaluate_performance(
                test_data,
                weights,
                eval_metric,
                calc_weights_kwargs.get("periods_per_year", PERIODS_PER_YEAR),
            )
            fold_perfs.append(perf)

        start_idx += step_size
        if start_idx + train_window + test_window > n_rows:
            break

    maximize = EVAL_METRICS[eval_metric]["maximize"]

    valid_perfs = [p for p in fold_perfs if not np.isnan(p)]
    if len(valid_perfs) == 0:
        agg_perf = -np.inf if maximize else np.inf
    else:
        if aggregator == "mean":
            agg_perf = float(np.mean(valid_perfs))
        elif aggregator == "median":
            agg_perf = float(np.median(valid_perfs))
        else:
            raise ValueError(
                f"Invalid aggregator: {aggregator}. Choose from ['mean', 'median']."
            )

    return agg_perf


### GRID SEARCH


def _tune_grid_search_tscv(
    returns: pd.DataFrame,
    param_dict: Dict[str, List[Union[float, str]]],
    eval_metric: str,
    aggregator: str,
    window_type: str,
    n_folds: int,
    train_window: int,
    test_window: int,
    step_size: int,
    **calc_weights_kwargs,
) -> Dict:
    """
    Perform a grid search over user-specified hyperparameter values and evaluate each
    combination via time-series cross-validation (TSCV).

    This function is invoked by the public `tune_regularization_tscv` method when
    `method="grid"`. It iterates over all parameter combinations in `param_grid`,
    merges them with default values from `PARAM_SPECS`, and calls `_run_tscv_for_params`
    to compute cross-validation scores.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns data.
    param_dict : Dict[str, List[Union[float, str]]]
        Hyperparameter candidates, where each key is a recognized parameter name
        (e.g., 'l1_reg') and its value is a list of possible numeric or categorical options.
    eval_metric : str
        The evaluation metric (e.g., mean_return, sharpe, volatility). Also used to decide
        whether we maximize or minimize scores.
    aggregator : str
        How to combine fold scores ('mean', 'median').
    window_type : str
        Type of cross-validation windowing ('rolling' or 'expanding').
    n_folds : int
        Number of TSCV folds (determined if not provided).
    train_window : int
        Number of periods in the training window.
    test_window : int
        Number of periods in the testing window.
    step_size : int
        How many periods to move forward after each fold. Defaults to test_window if None.
    **calc_weights_kwargs
        Additional keyword arguments passed through to `calc_weights`.
    """
    maximize = EVAL_METRICS[eval_metric]["maximize"]
    tuning_params = [p_name for p_name in PARAM_SPECS.keys() if p_name in param_dict]

    # hyperparameter values to use in grid search
    param_candidates = [param_dict[p_name] for p_name in tuning_params]

    # list of fold performances
    cv_results = {}
    best_score = -np.inf if maximize else np.inf
    best_params = {}

    for combo in product(*param_candidates):
        param_values = {}
        for p_name, p_val in zip(tuning_params, combo):
            param_values[p_name] = p_val

        # defaults for unspecified hyperparameters
        for p_name, spec in PARAM_SPECS.items():
            if p_name not in param_values:
                param_values[p_name] = spec["default"]

        score = _run_tscv_for_params(
            returns=returns,
            eval_metric=eval_metric,
            aggregator=aggregator,
            **param_values,
            window_type=window_type,
            n_folds=n_folds,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            **calc_weights_kwargs,
        )
        # store fold results in a dictionary for reference
        combo_dict = {p_name: param_values[p_name] for p_name in tuning_params}
        combo_key = tuple(sorted(combo_dict.items()))
        cv_results[combo_key] = score

        # check best
        if maximize:
            if score > best_score:
                best_score = score
                best_params = combo_dict
        else:
            if score < best_score:
                best_score = score
                best_params = combo_dict

    return {
        "best_params": best_params,
        "best_score": best_score,
        "cv_results": cv_results,
    }


### BAYES SEARCH


def _tune_bayes_search_tscv(
    returns: pd.DataFrame,
    param_dict: Dict[str, Union[Tuple[float, float], List[str]]],
    eval_metric: str,
    aggregator: str,
    window_type: str,
    n_folds: int,
    train_window: int,
    test_window: int,
    step_size: int,
    n_calls: int,
    random_state: int,
    **calc_weights_kwargs,
) -> Dict:
    """
    Perform a Bayesian optimization search over the hyperparameter space, evaluating
    each sampled combination via time-series cross-validation (TSCV).

    This function is invoked by the public `tune_regularization_tscv` method when
    `method="bayes"`. It uses scikit-optimize (`gp_minimize`) to adaptively sample
    parameter combinations from `param_dict`, merges them with default values from
    `PARAM_SPECS`, and calls `_run_tscv_for_params` to compute cross-validation scores.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns data.
    param_dict : Dict[str, Union[Tuple[float, float], List[str]]]
        Hyperparameter ranges or lists, where each key is a recognized parameter name
        (e.g., 'l1_reg') and its value is a numeric range (tuple) or a list
        for categorical parameters. This defines the space to be searched by Bayesian optimization.
    eval_metric : str
        The evaluation metric (e.g., mean_return, sharpe, volatility). Also used to decide
        whether we maximize or minimize scores.
    aggregator : str
        How to combine fold scores ('mean', 'median').
    window_type : str
        Type of cross-validation windowing ('rolling' or 'expanding').
    n_folds : int
        Number of TSCV folds (determined if not provided).
    train_window : int
        Number of periods in the training window.
    test_window : int
        Number of periods in the testing window.
    step_size : int
        How many periods to move forward after each fold. Defaults to test_window if None.
    n_calls : int
        The number of function evaluations (iterations) in the Bayesian optimization.
    random_state : int
        Random seed for reproducibility in the optimization procedure.
    **calc_weights_kwargs
        Additional keyword arguments passed to `calc_weights`.
    """
    dims = []
    dim_names = []
    cv_results = {}

    maximize = EVAL_METRICS[eval_metric]["maximize"]

    tuning_params = [p_name for p_name in PARAM_SPECS.keys() if p_name in param_dict]
    for p_name in tuning_params:
        spec = PARAM_SPECS[p_name]
        param_range = param_dict[p_name]
        if spec["type"] is Categorical:
            dims.append(spec["type"](param_range, name=p_name))
        else:
            dims.append(spec["type"](*param_range, name=p_name))
        dim_names.append(p_name)

    def objective_fun(params):
        param_values = dict(zip(dim_names, params))

        # defaults for unspecified hyperparameters
        for p_name, spec in PARAM_SPECS.items():
            if p_name not in param_values:
                param_values[p_name] = spec["default"]

        score = _run_tscv_for_params(
            returns=returns,
            eval_metric=eval_metric,
            aggregator=aggregator,
            **param_values,
            window_type=window_type,
            n_folds=n_folds,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            **calc_weights_kwargs,
        )
        combo_dict = {p_name: param_values[p_name] for p_name in tuning_params}
        combo_key = tuple(sorted(combo_dict.items()))
        cv_results[combo_key] = score

        return -score if maximize else score

    # run bayesian optimization
    result = gp_minimize(
        func=objective_fun, dimensions=dims, n_calls=n_calls, random_state=random_state
    )

    best_x = result.x
    best_raw_score = result.fun
    best_score = -best_raw_score if maximize else best_raw_score
    best_params = dict(zip(dim_names, best_x))

    return {
        "best_params": best_params,
        "best_score": best_score,
        "cv_results": cv_results,
        "skopt_result": result,
    }


def tune_regularization_tscv(
    returns: pd.DataFrame,
    param_dict: Dict[str, Union[List[Union[float, str]], Tuple[float, float]]],
    method: str = "grid",
    method_cofig: Dict = {},
    eval_metric: str = "sharpe",
    aggregator: str = "mean",
    window_type: str = "rolling",
    n_folds: int = None,
    train_window: int = None,
    test_window: int = None,
    step_size: int = None,
    **calc_weights_kwargs,
):
    """
    Perform time-series cross-validation (TSCV) to tune portfolio optimization regularization parameters.

    This function supports both grid search and Bayesian search for hyperparameter tuning, allowing users
    to optimize regularization parameters (e.g., `l1_reg`, `l2_reg`, `shrinkage_factor`, `shrinkage_target`)
    based on cross-validated performance metrics.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns data with rows representing time periods and columns representing assets.
    param_dict : Dict[str, Union[List[Union[float, str]], Tuple[float, float]]]
        Dictionary specifying the hyperparameters to tune. Each key must be a valid parameter name,
        and each value must be:
        - For grid search: a list of candidate values.
        - For Bayesian search: a tuple specifying the range for numerical parameters or a list for categorical parameters.
        Currently supported hyperparameters are `l1_reg`, `l2_reg`, `shrinkage_target`, and `shrinkage_factor`.
        Example for grid search:
            param_dict = {
                "l1_reg": [0.0, 0.01, 0.02],
                "l2_reg": [0.0, 0.01, 0.02],
                "shrinkage_factor": [0.0, 0.2, 0.5],
                "shrinkage_target": ["diagonal", "constant_correlation"]
            }
        Example for Bayesian search:
            param_dict = {
                "l1_reg": (0.0, 0.1),
                "l2_reg": (0.0, 0.1),
                "shrinkage_factor": (0.0, 1.0),
                "shrinkage_target": ["diagonal", "constant_correlation"]
            }
    method : str, default="grid"
        Optimization method to use. Must be one of:
        - "grid": Perform grid search over the parameter space.
        - "bayes": Perform Bayesian search using scikit-optimize.
    method_config : Dict, optional
        Additional method-specific configuration. Currently used only for Bayesian search:
        - "n_calls": Number of evaluations for Bayesian search (default: 20).
        - "random_state": Random seed for reproducibility (default: 42).
        Example:
            method_config = {
                "n_calls": 50,
                "random_state": 123
            }
    eval_metric : str, default="sharpe"
        The performance metric to optimize. Supported metrics include:
        - "mean_return": Maximize mean return.
        - "sharpe": Maximize Sharpe ratio.
        - "volatility": Minimize portfolio volatility.
    aggregator : str, default="mean"
        Aggregation method for cross-validation fold scores. Supported values:
        - "mean": Use the mean of fold scores.
        - "median": Use the median of fold scores.
    window_type : str, default="rolling"
        Type of cross-validation windowing. Supported values:
        - "rolling": Use rolling windows for train/test splits.
        - "expanding": Use expanding windows for train/test splits.
    n_folds : int, optional
        Number of cross-validation folds. If not specified, it will be calculated automatically based
        on the data size, train window, and test window.
    train_window : int, optional
        Number of periods in the training window. Required if not provided in percentage.
    test_window : int, optional
        Number of periods in the testing window. Required if not provided in percentage.
    step_size : int, optional
        Number of periods to step between consecutive train/test splits. Defaults to `test_window`.
    calc_weights_kwargs : dict, optional
        Additional keyword arguments to pass to the `calc_weights` function. Refer to `calc_weights` documentation
        for a full list of parameters.

    Returns
    -------
    Dict
        A dictionary containing:
        - "best_params": The best-performing hyperparameter combination.
        - "best_score": The best cross-validated performance score.
        - "cv_results": A dictionary of all hyperparameter combinations and their respective scores.

    Raises
    ------
    ValueError
        If any of the following conditions are met:
        - Invalid keys in `param_dict`.
        - `param_dict` does not specify at least one valid hyperparameter with a non-empty range.
        - Tuning parameters are specified in `calc_weights_kwargs` instead of `param_dict`.
        - Invalid `eval_metric` is provided.
        - Invalid `method` is specified.
    TypeError
        If `eval_metric` is not a valid evaluation metric.
    """

    # validate that all keys in param_dict are valid
    invalid_keys = [k for k in param_dict if k not in PARAM_SPECS]
    if invalid_keys:
        valid_keys = ", ".join(PARAM_SPECS.keys())
        raise ValueError(
            f"Invalid keys in 'param_dict': {', '.join(invalid_keys)}. Valid keys are: {valid_keys}."
        )

    # validate that no hyperparameters were specified in the calc_weights arguments
    misconfigured_params = [k for k in calc_weights_kwargs if k in PARAM_SPECS]
    if misconfigured_params:
        raise ValueError(
            f"The following tuning parameters must be configured in 'param_dict', not in 'calc_weights_kwargs': "
            f"{', '.join(misconfigured_params)}. See documentation for details."
        )

    # validate that at least one hyperparameter is specified for tuning
    tuning_params = [
        p for p, values in param_dict.items() if p in PARAM_SPECS and len(values) > 0
    ]
    if not tuning_params:
        raise ValueError(
            "At least one hyperparameter must be specified with a non-empty list or tuple in 'param_dict'. "
            f"Supported hyperparameters are: {', '.join(PARAM_SPECS.keys())}."
        )

    # validate evaluation metric
    if eval_metric not in EVAL_METRICS:
        raise ValueError(
            f"Invalid eval_metric: '{eval_metric}'. Valid options are: {', '.join(EVAL_METRICS)}."
        )

    if method == "grid":
        return _tune_grid_search_tscv(
            returns=returns,
            param_dict=param_dict,
            eval_metric=eval_metric,
            aggregator=aggregator,
            window_type=window_type,
            n_folds=n_folds,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            **calc_weights_kwargs,
        )
    elif method == "bayes":
        return _tune_bayes_search_tscv(
            returns=returns,
            param_dict=param_dict,
            eval_metric=eval_metric,
            aggregator=aggregator,
            window_type=window_type,
            n_folds=n_folds,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            n_calls=method_cofig.get("n_calls", 20),
            random_state=method_cofig.get("random_state", 42),
            **calc_weights_kwargs,
        )
    else:
        raise ValueError("method must be 'grid' or 'bayes'.")
