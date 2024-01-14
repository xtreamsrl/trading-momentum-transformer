# HP_HIDDEN_LAYER_SIZE = [5, 10, 20, 40, 80, 160]
# HP_DROPOUT_RATE = [0.1, 0.2, 0.3, 0.4, 0.5]
# HP_MINIBATCH_SIZE = [64, 128, 256]
# HP_LEARNING_RATE = [1e-4, 1e-3, 1e-2, 1e-1]
# HP_MAX_GRADIENT_NORM = [0.01, 1.0, 100.0]
# RANDOM_SEARCH_ALGORITHM = kt.RandomSearch

from typing import Any


def get_hp_grid_by_test_start(test_start: int) -> dict[str, Any]:
    return {
        2016: {
            "hidden_layer_size": 5,
            "dropout_rate": 0.5,
            "max_gradient_norm": 100.0,
            "learning_rate": 0.01,
            "batch_size": 64
        },
        2017: {
            "hidden_layer_size": 80,
            "dropout_rate": 0.3,
            "max_gradient_norm": 0.01,
            "learning_rate": 0.001,
            "batch_size": 64
        },
        2018: {
            "hidden_layer_size": 160,
            "dropout_rate": 0.3,
            "max_gradient_norm": 0.01,
            "learning_rate": 0.001,
            "batch_size": 256
        },
        2019: {
            "hidden_layer_size": 10,
            "dropout_rate": 0.1,
            "max_gradient_norm": 1.0,
            "learning_rate": 0.001,
            "batch_size": 128
        },
        2020: {
            "hidden_layer_size": 10,
            "dropout_rate": 0.2,
            "max_gradient_norm": 0.01,
            "learning_rate": 0.01,
            "batch_size": 64
        }
    }.get(test_start)
