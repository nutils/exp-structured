import numpy as np


def create_grid(grid_lines, element_size, sample_density=0.1, merge_scale=0.01):
    indices = np.argsort(grid_lines)
    where = np.empty_like(indices)

    # initialize left-most grid line
    i = indices[0]
    grid = [grid_lines[i]]
    where[i] = 0

    # stack additional grid lines
    for i in indices[1:]:
        # sample element size distribution via bisection of the interval
        intervals = [
            grid[-1],
            element_size((grid[-1] + grid_lines[i]) / 2),
            grid_lines[i],
        ]
        stack = [0]
        while stack:
            k = stack.pop()  # start index of x0, e, x1 triplet
            x0, e, x1 = intervals[k : k + 3]
            w = (x1 - x0) / 3
            if w > sample_density * e:
                intervals.insert(k + 1, element_size(x0 + 0.5 * w))
                intervals.insert(k + 2, x0 + w)
                intervals.insert(k + 4, x1 - w)
                intervals.insert(k + 5, element_size(x1 - 0.5 * w))
                stack.extend([k, k + 2, k + 4])
        intervals = np.stack(intervals)

        # ξ dimensionless, gridlines at 0, 1, 2
        # construct x(ξ) such that dx/dξ = e(x)
        # dξ = dx / e(x) => ξ = ∫_0^ξ dξ = ∫_x0^x(ξ) (1 / e(x)) dx
        x = intervals[0::2]
        e = intervals[1::2]
        ξ = np.cumsum([0.0, *np.diff(x) / e])

        # ξ[-1] is the grid coordinate at which x matches grid_lines[i]; if it is very
        # close to zero we merge with the existing mesh line
        if ξ[-1] > merge_scale:
            # ξgrid is a uniform division of the [0, ξ[-1]] interval in round(ξ[-1])
            # steps, excluding the leading 0 to avoid repetition
            ξgrid = (
                np.linspace(0, 1, 1 + max(1, round(ξ[-1])))[1:] * ξ[-1]
            )  # approximately 1, 2, 3, etc
            # extend the grid with x coordinates matching a uniform sampling of ξ
            grid.extend(np.interp(ξgrid, ξ, x))

        where[i] = len(grid) - 1

    return np.stack(grid), where
