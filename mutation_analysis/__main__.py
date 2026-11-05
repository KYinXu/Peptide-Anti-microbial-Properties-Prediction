"""Allow `python -m mutation_analysis` to run mutation-robustness analysis."""

from .run_mutation_analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
