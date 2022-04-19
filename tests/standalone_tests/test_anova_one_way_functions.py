import numpy as np

from mipengine.algorithms.one_way_anova import get_ci_info


def test_min_max_ci():
    means = [2, 3, 4, 5]
    sample_stds = [1, 2, 1, 1]
    categories = ["A", "B", "C", "D"]
    group_stats_index = ["B", "A", "C", "D"]
    ci_info = get_ci_info(means, sample_stds, categories, group_stats_index)
    ci_info_exp = {
        "m-s": {"A": 1, "B": 1, "C": 3, "D": 4},
        "m+s": {"A": 3, "B": 5, "C": 5, "D": 6},
    }
    for key, e_val in ci_info_exp.items():
        r_val = ci_info[key]
        assert e_val == r_val or np.isclose(e_val, r_val)


test_min_max_ci()
