Getting Started
===============

Installation
------------
If you haven’t already installed the package, you can install it using:

.. code-block:: console

   pip install portfolio-management

Overview
--------
This package contains modules for portfolio analysis, construction, risk management,
and more. It leverages pandas, numpy, statsmodels, arch, and more.

Usage Example
-------------
Here’s a brief example of how you might import and use a function:

.. code-block:: python

   import pandas as pd
   from portfolio_management.analysis import calc_cross_section_regression

   # Suppose 'returns_df' is a DataFrame of returns
   # and 'factors_df' is a DataFrame of factors
   result = calc_cross_section_regression(returns_df, factors_df)
   print(result)

That’s it! For more details, see the API reference in the next sections.
