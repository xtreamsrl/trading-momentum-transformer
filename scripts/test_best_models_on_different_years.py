import logging
import os

import tensorflow as tf

from mom_trans.backtest import run_single_window
from settings.default import QUANDL_TICKERS
from settings.hp_grid import get_hp_grid_by_test_start


def main():
    experiment_name = "experiment_quandl_100assets_tft_cpnone_len63_notime_div_v1_different_test_years"
    features_file_path = os.path.join('..', "data", "quandl_cpd_nonelbw.csv")
    intervals = [(1990, y, 2022) for y in range(2016, 2021)]
    changepoint_lbws = None
    asset_class_dictionary = dict(zip(QUANDL_TICKERS, ["COMB"] * len(QUANDL_TICKERS)))
    params = {
        "architecture": "TFT",
        "total_time_steps": 63,
        "early_stopping_patience": 25,
        "multiprocessing_workers": 32,
        "num_epochs": 300,
        "fill_blank_dates": False,
        "split_tickers_individually": True,
        "random_search_iterations": 1,  # EF: we are going to fix the hyperparameters
        "evaluate_diversified_val_sharpe": True,
        "train_valid_ratio": 0.90,
        "time_features": False,
        "force_output_sharpe_length": None,
    }

    tf.config.set_visible_devices([], 'GPU')  # disable GPU as the batch size is too small
    for interval in intervals:
        hp_minibatch_size = [get_hp_grid_by_test_start(interval[1])["batch_size"]]
        params["hidden_layer_size"] = get_hp_grid_by_test_start(interval[1])["hidden_layer_size"]
        params["dropout_rate"] = get_hp_grid_by_test_start(interval[1])["dropout_rate"]
        params["max_gradient_norm"] = get_hp_grid_by_test_start(interval[1])["max_gradient_norm"]
        params["learning_rate"] = get_hp_grid_by_test_start(interval[1])["learning_rate"]
        run_single_window(
            experiment_name,
            features_file_path,
            interval,
            params,
            changepoint_lbws,
            asset_class_dictionary=asset_class_dictionary,
            hp_minibatch_size=hp_minibatch_size,
        )


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(module)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    main()
