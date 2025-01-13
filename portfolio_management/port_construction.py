import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Union
from enum import Enum


# Default for number of periods per year, used for annualizing returns and variances
# Defaults to 12 for monthly data but can be adjusted for other frequencies (e.g., 252 for daily, 4 for quarterly)
PERIODS_PER_YEAR = 12


class ObjectiveType(Enum):
    """Enum of supported portfolio optimization objective types."""

    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_RETURN = "max_return"
    MEAN_VARIANCE = "mean_variance"


### HELPER SUBROUTINES


def _build_objective(
    objective_type: ObjectiveType,
    x: cp.Variable,
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_tolerance: float,
) -> cp.Expression:
    """
    Construct the cvxpy-compatible objective expression for portfolio optimization.

    Parameters
    ----------
    objective_type : ObjectiveType
        The optimization objective, as defined in the `ObjectiveType` enum.
    x : cp.Variable
        Portfolio weight vector (cvxpy variable).
    mu : np.ndarray
        Expected returns vector.
    Sigma : np.ndarray
        Covariance matrix of asset returns.
    risk_tolerance : float
        Risk tolerance parameter for mean-variance optimization.

    Returns
    -------
    cp.Expression
        The cvxpy-compatible objective expression for the optimization problem.

    Raises
    ------
    ValueError
        If the provided objective type is not implemented.
    """
    if objective_type == ObjectiveType.MEAN_VARIANCE:
        # minimize 0.5 * x^T Sigma x - gamma * mu^T x
        return 0.5 * cp.quad_form(x, Sigma) - risk_tolerance * (mu @ x)

    elif objective_type in {ObjectiveType.MIN_VARIANCE, ObjectiveType.MAX_SHARPE}:
        # minimize x^T Sigma x
        return cp.quad_form(x, Sigma)

    elif objective_type == ObjectiveType.MAX_RETURN:
        # maximize x^T mu == minimize -x^T mu
        return -mu @ x

    raise NotImplementedError(f"Objective type '{objective_type}' is not implemented.")


def _add_constraints(
    objective_type: ObjectiveType,
    x: cp.Variable,
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float = None,
    target_variance: float = None,
    strict_targets: bool = False,
    long_only: bool = False,
) -> list:
    """
    Build and return portfolio optimization constraints based on specified parameters.

    Parameters
    ----------
    objective_type : ObjectiveType
        The optimization objective, as defined in the `ObjectiveType` enum.
    x : cp.Variable
        Portfolio weight vector (cvxpy variable).
    mu : np.ndarray
        Expected returns vector.
    Sigma : np.ndarray
        Covariance matrix of asset returns.
    target_return : float, optional
        Target portfolio return constraint. If provided, enforces a portfolio return.
    target_variance : float, optional
        Target portfolio variance constraint. If provided, enforces a portfolio variance.
    strict_targets : bool, optional
        If True, enforces strict equality (`==`) for target_return and target_variance constraints.
        If False, uses inequalities (`>=` for return and `<=` for variance).
    long_only : bool, optional
        If True, enforces non-negative weights (long-only portfolio).

    Returns
    -------
    list
        A list of cvxpy constraints for the optimization problem.
    """
    constraints = []

    # sum weights to 1
    if objective_type == ObjectiveType.MAX_SHARPE:
        constraints.append(mu @ x == 1)
    else:
        constraints.append(cp.sum(x) == 1)

    if long_only:
        constraints.append(x >= 0)

    if target_return is not None:
        if strict_targets:
            constraints.append(mu @ x == target_return)
        else:
            constraints.append(mu @ x >= target_return)

    if target_variance is not None:
        if strict_targets:
            constraints.append(cp.quad_form(x, Sigma) == target_variance)
        else:
            constraints.append(cp.quad_form(x, Sigma) <= target_variance)

    return constraints


def _add_regularization_to_objective(
    objective_expr: cp.Expression,
    x: cp.Variable,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
) -> cp.Expression:
    """
    Adds L1 (Lasso) and L2 (Ridge) regularization terms to the optimization objective.

    Parameters
    ----------
    objective_expr : cp.Expression
        The existing cvxpy objective expression to which regularization terms will be added.
    x : cp.Variable
        Portfolio weight vector (cvxpy variable).
    l1_reg : float, optional
        Coefficient for L1 regularization (Lasso). Defaults to 0.0.
    l2_reg : float, optional
        Coefficient for L2 regularization (Ridge). Defaults to 0.0.

    Returns
    -------
    cp.Expression
        The modified objective expression with added regularization terms.
    """
    # L1 (Lasso) penalty
    if l1_reg > 0:
        objective_expr += l1_reg * cp.norm1(x)

    # L2 (Ridge) penalty
    if l2_reg > 0:
        objective_expr += l2_reg * cp.sum_squares(x)

    return objective_expr


def _shrink_covariance(
    Sigma: np.ndarray,
    target: Union[str, np.ndarray] = "diagonal",
    shrinkage_factor: float = None,
) -> np.ndarray:
    r"""
    Generalized shrinkage function for covariance matrix.

    This function applies shrinkage to the input covariance matrix \(\Sigma\) by blending it
    with a specified target matrix using a shrinkage factor \(\alpha\). The formula is:

        \(\Sigma_{\text{shrunk}} = \alpha \cdot \text{target\_matrix} + (1 - \alpha) \cdot \Sigma\)

    Parameters:
    ----------
    Sigma : np.ndarray
        Covariance matrix.
    target : Union[str, np.ndarray]
        Target covariance matrix. Options:
            - "diagonal": Diagonal covariance matrix from variances.
            - "constant_correlation": Shrinks to a constant correlation matrix.
            - Custom matrix: Provide an explicit target matrix as a numpy array.
    shrinkage_factor : float, optional
        Shrinkage factor (\(\alpha\)). If None, the optimal shrinkage is calculated using
        the Ledoit-Wolf (2003) method.

    Returns:
    -------
    np.ndarray
        Shrunk covariance matrix.
    """
    # define the target matrix
    if target == "diagonal":
        # use diagonal matrix with variances
        target_matrix = np.diag(np.diag(Sigma))
    elif target == "constant_correlation":
        # construct a constant correlation matrix
        std_devs = np.sqrt(np.diag(Sigma))
        avg_corr = np.mean(Sigma / np.outer(std_devs, std_devs))
        target_matrix = avg_corr * np.outer(std_devs, std_devs)
    elif isinstance(target, np.ndarray):
        # custom target matrix
        target_matrix = target
    else:
        raise ValueError(
            "Invalid target specified. Use 'diagonal', 'constant_correlation', or a custom matrix."
        )

    # if shrinkage factor not provided, use Ledoit-Wolf formula to compute optimal shrinkage_factor (alpha)
    if shrinkage_factor is None:
        diff = Sigma - target_matrix
        numerator = np.sum(diff**2)
        denominator = np.sum((Sigma - np.mean(Sigma, axis=0)) ** 2)
        shrinkage_factor = numerator / (numerator + denominator)

    # apply shrinkage using Sigma_shrunk = alpha * target_matrix + (1 - alpha) * Sigma
    shrunk_cov = shrinkage_factor * target_matrix + (1 - shrinkage_factor) * Sigma

    return shrunk_cov


def _validate_and_map_inputs(
    objective_type: Union[str, ObjectiveType],
    target_return: float,
    target_variance: float,
    risk_tolerance: float,
    shrinkage_target: Union[str, np.ndarray],
    shrinkage_factor: float,
) -> ObjectiveType:
    """
    Validates and maps inputs for the calc_weights function.

    Parameters
    ----------
    objective_type : Union[str, ObjectiveType]
        Type of optimization objective. Can be passed as a string or an ObjectiveType enum.
    target_return : float
        Enforces the portfolio return. Used with 'min_variance' or 'mean_variance'.
    target_variance : float
        Enforces the portfolio variance. Used with 'max_return' or 'mean_variance'.
    risk_tolerance : float
        Risk tolerance parameter (gamma) for the 'mean_variance' objective.
    shrinkage_target : Union[str, np.ndarray]
        Shrinkage target for covariance matrix.
    shrinkage_factor : float
        Shrinkage intensity (alpha), between 0 and 1.
    l1_reg : float
        L1 regularization penalty (Lasso).
    long_only : bool
        Whether portfolio weights are constrained to be non-negative.

    Returns
    -------
    ObjectiveType
        Validated and mapped ObjectiveType enum.
    """

    # map and validate objective_type
    if isinstance(objective_type, str):
        try:
            objective_type = ObjectiveType(objective_type)
        except ValueError:
            raise ValueError(
                f"Invalid objective_type '{objective_type}'. Valid options are: {", ".join([e.value for e in ObjectiveType])}."
            )
    elif not isinstance(objective_type, ObjectiveType):
        raise TypeError(
            f"objective_type must be of type str or ObjectiveType, not {type(objective_type)}."
        )

    # shrinkage checks
    if shrinkage_factor is not None:
        if not (0 <= shrinkage_factor <= 1):
            raise ValueError("`shrinkage factor` must be between 0 and 1, inclusive.")
        if shrinkage_target is None:
            raise ValueError(
                "`shrinkage_target` must be specified when providing a `shrinkage_factor`."
            )

    # objective checks
    if objective_type != ObjectiveType.MEAN_VARIANCE and risk_tolerance is not None:
        raise ValueError(
            "`risk_tolerance` can only be used when objective_type='mean_variance'."
        )
    if objective_type == ObjectiveType.MEAN_VARIANCE and risk_tolerance is None:
        raise ValueError(
            "`risk_tolerance` must be specified when objective_type='mean_variance'."
        )

    if objective_type == ObjectiveType.MAX_SHARPE and (
        target_return is not None or target_variance is not None
    ):
        raise ValueError(
            "Cannot specify `target_return` or `target_variance` when objective_type='max_sharpe'. "
            "If you want a specific return, use 'min_variance' with `target_return`. "
            "If you want a specific variance, use 'max_return' with `target_variance`, or consider 'mean_variance'."
        )

    return objective_type


### MASTER FUNCTION


def calc_weights(
    returns: pd.DataFrame,
    objective_type: Union[str, ObjectiveType] = "max_sharpe",
    target_return: float = None,
    target_variance: float = None,
    strict_targets: bool = False,
    risk_tolerance: float = None,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
    shrinkage_target: Union[str, np.ndarray] = None,
    shrinkage_factor: float = None,
    long_only: bool = False,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """
    Calculate portfolio weights using a flexible framework that supports multiple objectives, constraints, and regularization methods.
    Portfolio weights are constrained to sum to 1.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset (columns=assets).
    objective_type : Union[str, ObjectiveType]
        Optimization objective type. Can be a string (e.g., 'min_variance', 'max_sharpe')
        or an ObjectiveType enum. Internally, strings are mapped to ObjectiveType enums.
    target_return : float, optional
        If set, enforces portfolio return.
    target_variance : float, optional
        If set, enforces portfolio variance.
    strict_targets : bool, optional
        If True, enforces strict equality (`==`) for target_return and target_variance constraints.
        If False, uses inequalities (`>=` for return and `<=` for variance).
    risk_tolerance : float, optional
        Risk-return trade-off parameter (gamma) for the mean-variance objective.
        Only used when objective_type is "mean_variance".
    l1_reg : float, optional
        L1 penalty factor (Lasso), penalizing the absolute sum of weights.
    l2_reg : float, optional
        L2 penalty factor (Ridge), penalizing the squared sum of weights.
    shrinkage_target : Union[str, np.ndarray], optional
        The shrinkage target for the covariance matrix. Options:
            - "diagonal": Diagonal covariance matrix from variances.
            - "constant_correlation": Shrinks to a constant correlation matrix.
            - Custom matrix: Provide an explicit target matrix as a numpy array.
            - None: No shrinkage is applied.
    shrinkage_factor : float, optional
        Shrinkage intensity (alpha), a value between 0 and 1:
        - 0: No shrinkage (uses the raw sample covariance matrix).
        - 1: Fully shrinks to the specified target matrix.
        If None, the optimal shrinkage is automatically calculated using the Ledoit-Wolf method.
        If explicitly set, `shrinkage_target` must also be provided.
    long_only : bool, optional
        If True, enforces x >= 0, making the portfolio long-only.
    periods_per_year : int, optional
        Used to annualize return and covariance estimates, e.g., 12 for monthly data.

    Returns
    -------
    pd.Series
        Portfolio weights, indexed by returns.columns
    """
    # validate inputs
    objective_type = _validate_and_map_inputs(
        objective_type,
        target_return,
        target_variance,
        risk_tolerance,
        shrinkage_target,
        shrinkage_factor,
    )

    # generate inputs for portfolio optimization
    Sigma = returns.cov().values * periods_per_year
    mu = returns.mean().values * periods_per_year
    n_assets = len(returns.columns)
    x = cp.Variable(n_assets)

    # apply shrinkage to covariance matrix
    if shrinkage_target is not None:
        Sigma = _shrink_covariance(
            Sigma, target=shrinkage_target, shrinkage_factor=shrinkage_factor
        )

    # build objective
    objective_expr = _build_objective(objective_type, x, mu, Sigma, risk_tolerance)

    # add L1/L2 regularization
    objective_expr = _add_regularization_to_objective(objective_expr, x, l1_reg, l2_reg)

    # build constraints
    constraints = _add_constraints(
        objective_type,
        x,
        mu,
        Sigma,
        target_return=target_return,
        target_variance=target_variance,
        strict_targets=strict_targets,
        long_only=long_only,
    )

    # solve objective function with constraints
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    problem.solve()

    # check solution converged
    if x.value is None:
        raise ValueError(f"Solver did not converge for objective={objective_type}.")

    weights = x.value

    # for max sharpe, rescale weights to a sum of 1
    if objective_type == ObjectiveType.MAX_SHARPE:
        sum_weights = np.sum(weights)
        if sum_weights <= 0:
            raise ValueError(
                "The solution has non-positive sum; cannot normalize to sum=1."
            )

        weights /= sum_weights

    return pd.Series(weights, index=returns.columns, name="Weights")


### PUBLIC HELPERS


def scale_weights(
    returns: pd.DataFrame,
    weights: pd.Series,
    target_return: float = None,
    target_variance: float = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """
    Scale portfolio weights to achieve a target return or target variance.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns data for assets.
    weights : pd.Series
        Portfolio weights.
    target_return : float, optional
        Target annualized portfolio return.
    target_variance : float, optional
        Target annualized portfolio variance.
    periods_per_year : int, optional
        Number of periods in a year for annualization.

    Returns
    -------
    pd.Series
        Scaled portfolio weights to meet the specified target.

    Raises
    ------
    ValueError
        If both `target_return` and `target_variance` are specified.
    """
    # check that only one target is specified
    if target_return is not None and target_variance is not None:
        raise ValueError(
            "Only one target can be specified: either `target_return` or `target_variance`, not both."
        )

    scaled_weights = weights.copy()

    # scale weights to achieve target return
    if target_return is not None:
        portfolio_returns = returns @ scaled_weights
        mean_return = portfolio_returns.mean()
        if periods_per_year:
            mean_return *= periods_per_year
        scaler = target_return / mean_return
        scaled_weights *= scaler

    # scale weights to achieve target variance
    if target_variance is not None:
        portfolio_variance = (returns @ scaled_weights).var()
        if periods_per_year:
            portfolio_variance *= periods_per_year
        scaler = (target_variance / portfolio_variance) ** 0.5
        scaled_weights *= scaler

    return scaled_weights


### BASIC METHODS - SIMPLE USER ENTRY POINTS


def calc_equal_weights(
    returns: pd.DataFrame,
    target_return: float = None,
    target_variance: float = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """
    Calculate equal-weighted weights, optionally scaled to meet a target return or variance.
    """
    n_assets = len(returns.columns)
    weights = pd.Series(
        np.ones(n_assets) / n_assets, index=returns.columns, name="Weights"
    )

    # scale weights to target return or target variance if specified
    weights = scale_weights(
        returns, weights, target_return, target_variance, periods_per_year
    )

    return weights


def calc_min_variance_weights(returns: pd.DataFrame, **calc_weight_kwargs) -> pd.Series:
    """
    Calculate weights for the minimum-variance portfolio.
    """
    if "objective_type" in calc_weight_kwargs:
        raise ValueError(
            "calc_min_variance_weights sets objective_type=MIN_VARIANCE. Please remove 'objective_type' from calc_weight_kwargs or call calc_weights directly."
        )

    return calc_weights(
        returns, objective_type=ObjectiveType.MIN_VARIANCE, **calc_weight_kwargs
    )


def calc_tangency_weights(returns: pd.DataFrame, **calc_weight_kwargs) -> pd.Series:
    """
    Calculate weights for the tangency (maximum Sharpe ratio) portfolio.
    """
    if "objective_type" in calc_weight_kwargs:
        raise ValueError(
            "calc_tangency_weights sets objective_type=MAX_SHARPE. Please remove 'objective_type' from calc_weight_kwargs or call calc_weights directly."
        )

    return calc_weights(
        returns, objective_type=ObjectiveType.MAX_SHARPE, **calc_weight_kwargs
    )


def calc_target_return_weights(
    returns: pd.DataFrame, target_return: float, **calc_weight_kwargs
) -> pd.Series:
    """
    Calculate weights for the portfolio minimizing variance subject to a target return.
    """
    if "objective_type" in calc_weight_kwargs:
        raise ValueError(
            "calc_target_return_weights sets objective_type=MIN_VARIANCE. Please remove 'objective_type' from calc_weight_kwargs or call calc_weights directly."
        )

    if "target_variance" in calc_weight_kwargs:
        raise ValueError(
            "calc_target_return_weights is for a target return, not a target variance. Please remove 'target_variance' or use `calc_target_variance_weights`."
        )

    return calc_weights(
        returns,
        objective_type=ObjectiveType.MIN_VARIANCE,
        target_return=target_return,
        **calc_weight_kwargs,
    )


def calc_target_variance_weights(
    returns: pd.DataFrame, target_variance: float, **calc_weight_kwargs
) -> pd.Series:
    """
    Calculate weights for the portfolio maximizing return subject to a target variance.
    """
    if "objective_type" in calc_weight_kwargs:
        raise ValueError(
            "calc_target_variance_weights sets objective_type=MAX_RETURN. Please remove 'objective_type' from calc_weight_kwargs or call calc_weights directly."
        )

    if "target_return" in calc_weight_kwargs:
        raise ValueError(
            "calc_target_variance_weights is for a target variance, not a target return. Please remove 'target_return' or use `calc_target_return_weights`."
        )

    return calc_weights(
        returns,
        objective_type=ObjectiveType.MAX_RETURN,
        target_variance=target_variance,
        **calc_weight_kwargs,
    )


def calc_mean_variance_weights(
    returns: pd.DataFrame, risk_tolerance: float, **calc_weight_kwargs
) -> pd.Series:
    """
    Calculate weights for the portfolio using mean-variance optimization.
    """
    if "objective_type" in calc_weight_kwargs:
        raise ValueError(
            "calc_mean_variance_weights sets objective_type=MEAN_VARIANCE. Please remove 'objective_type' from calc_weight_kwargs or call calc_weights directly."
        )

    return calc_weights(
        returns,
        objective_type=ObjectiveType.MEAN_VARIANCE,
        risk_tolerance=risk_tolerance,
        **calc_weight_kwargs,
    )


def calc_risk_parity_weights(
    returns: pd.DataFrame,
    target_return: float = None,
    target_variance: float = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """
    Calculate risk parity weights, optionally scaled to meet a target return or variance.
    """

    weights = pd.Series(
        [1 / returns[asset].var() for asset in returns.columns],
        index=returns.columns,
        name="Weights",
    )

    # scale weights to target return or target variance if specified
    weights = scale_weights(
        returns, weights, target_return, target_variance, periods_per_year
    )

    return weights
