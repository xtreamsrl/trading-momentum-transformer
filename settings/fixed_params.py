MODEL_PARAMS = {
    "architecture": "TFT",
    "total_time_steps": 252,
    "early_stopping_patience": 25,
    "multiprocessing_workers": 32,
    "num_epochs": 300,
    "fill_blank_dates": False,
    "split_tickers_individually": True,
    "random_search_iterations": 25, # EF: original was 50, but it takes too long
    "evaluate_diversified_val_sharpe": True,
    "train_valid_ratio": 0.90,
    "time_features": False,
    "force_output_sharpe_length": 0,
}
