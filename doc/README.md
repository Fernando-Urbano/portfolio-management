# Documentation Guide

This guide explains how to **create docstrings** in NumPy style and **build** the Sphinx HTML documentation for this project.

---

## 1. Writing Docstrings (NumPy Style)

Our documentation uses **NumPy-style docstrings**, which require the [Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension in Sphinx. Below is a **minimal example** of how to document a function:

```python
def my_function(param1: str, param2: int = 0) -> bool:
    """
    A short summary of what the function does.

    A longer description can follow on a new line, describing in detail
    the function's purpose, arguments, behavior, and so on.

    Parameters
    ----------
    param1 : str
        Description of the first parameter.
    param2 : int, optional
        Description of the second parameter (with default=0).

    Returns
    -------
    bool
        Explanation of what the function returns or indicates.

    Notes
    -----
    You can add optional sections like Examples, Warnings, References, etc.

    Examples
    --------
    >>> result = my_function("hello", 5)
    >>> assert result is True
    """
    return True
```

### Sections You May Include

- **Parameters**: List input parameters.
- **Returns** or **Yields**: Describe returned values or yielded values (for generators).
- **Raises**: Mention any exceptions.
- **Notes**, **Examples**, **References**, etc. (optional).

For **classes**, you can add a docstring under the class definition and under each method:

```python
class MyClass:
    """
    A short summary of what this class does.

    Parameters
    ----------
    init_param : float
        Description of the initialization parameter.
    """

    def __init__(self, init_param: float):
        self.init_param = init_param

    def method_one(self, x: int) -> float:
        """
        Brief description of the method.

        Parameters
        ----------
        x : int
            Some integer input.

        Returns
        -------
        float
            A floating-point output.
        """
        return self.init_param * x
```

---

## 2. Building the HTML Documentation

Below is a step-by-step guide to **build** the Sphinx docs locally:

1. **Activate your virtual environment (if applicable)**:

   ```bash
   conda activate finm
   # or, if you're using a standard venv
   source venv/bin/activate
   ```

2. **Install dependencies**:

   Make sure you have [Sphinx](https://pypi.org/project/Sphinx/) installed, along with the **PyData Sphinx Theme** and **Napoleon**. For example:

   ```bash
   pip install sphinx pydata-sphinx-theme
   # "napoleon" is included in sphinx.ext.napoleon
   ```

3. **Navigate to the `doc/` folder**:

   ```bash
   cd doc
   ```

4. **Build the documentation**:

   - **macOS/Linux**:
     ```bash
     make html
     ```
   - **Windows**:
     ```bash
     make.bat html
     ```

5. **View the resulting docs**:

   - The build process creates the HTML files in `doc/build/html/`.
   - **Open** `doc/build/html/index.html` in your web browser.
     - You can either **double-click** `index.html` in Finder/Explorer,
     - **or** run a command (macOS example):
       ```bash
       open doc/build/html/index.html
       ```
     - On Windows:
       ```bash
       start doc\\build\\html\\index.html
       ```
     - On Linux:
       ```bash
       xdg-open doc/build/html/index.html
       ```

### Using a Local HTTP Server (Optional)

Instead of opening `index.html` directly, you can serve the docs with a local HTTP server for a smoother experience:

```bash
cd doc/build/html
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 3. Updating Docstrings & Rebuilding

1. Modify or add your docstrings in the `.py` modules (e.g., `analysis.py`, `risk.py`, etc.).
2. **Re-run** the build steps:
   ```bash
   cd doc
   make clean
   make html
   ```
3. Open `doc/build/html/index.html` to see updated docs.

---

## 4. Common Issues

- **`Theme error: no theme named 'pydata_sphinx_theme' found`**
  Make sure you installed the theme in your current Python environment:
  ```bash
  pip install pydata-sphinx-theme
  ```

- **Missing `.nojekyll`**
  If you see a warning:
  ```text
  WARNING: html_extra_path entry '.nojekyll' does not exist
  ```
  Create an empty `.nojekyll` file in `doc/source/`:
  ```bash
  touch doc/source/.nojekyll
  ```

- **Tabs vs Spaces in Makefile**
  The `make` command requires **tabs** for indentation in the Makefile. If you see `missing separator.  Stop.`, ensure each recipe line is indented with a **tab** (ASCII 0x09).

---

## 5. Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/):
  - Guides to advanced features, like cross-referencing, autodoc for classes, etc.
- [Napoleon (NumPy/Google docstring style)](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html):
  - Explains the recognized sections and how to format them.
- [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/):
  - Documentation on theme customization.

---

**Happy Documenting!**
