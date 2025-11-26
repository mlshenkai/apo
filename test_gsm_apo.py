from apo.pipeline import run_apo_pipeline


run_apo_pipeline(
        task="gsm8k",
        n_rounds=1,
        train_path="./local_datasets/gsm8k_small/train.jsonl",
        test_path="./local_datasets/gsm8k_small/test.jsonl",
        debug=False,
    )