# Experiments

:warning: This directory and its contents are experimental and may be removed from `ukat` at any point.

## Difference between experiments and tutorials

- `tutorials` directory: contains code strictly for demonstration main features of `ukat`.
- `experiments` directory: contains code (and associated data if relevant) that supports decisions taken in the implementation of `ukat` tools, that doesn't need to be part of the `ukat` package but which we want to make reproducible.

## List of experiments

### `philips_b0_validate_phase_calculation`

- Purpose: Demonstrate calculation of magnitude and phase from real and imaginary data that matches magnitude and phase data calculated at the scanner.
- Data: 3 B0 mapping datasets with different shims. Each dataset contains real, imaginary, magnitude and phase data.
- _Note for @fnery: The files in the data directory are a subset of dataset 20200820 (see log:20200823)_.

### `unwrapping`
- Purpose: Compare unwrapping methods: `unwrap_phase` from the `restoration` module from `scikit-image` and `PRELUDE` from FSL.
- See more details [here](./unwrapping/README.md).
