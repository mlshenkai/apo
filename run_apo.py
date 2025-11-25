# run_apo.py
import argparse
from apo.pipeline import run_apo_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run APO (ELPO) pipeline on a task.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["liar", "bbh", "ethos", "arsarcasm", "wsc", "gsm8k"],
        help="Task name."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of optimization rounds (default: from .env DEFAULT_ROUNDS)."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Path to train.jsonl (if None, use default under local_datasets/)."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Path to test.jsonl (if None, use default under local_datasets/)."
    )
    args = parser.parse_args()

    run_apo_pipeline(
        task=args.task,
        n_rounds=args.rounds,
        train_path=args.train_path,
        test_path=args.test_path,
    )


if __name__ == "__main__":
    main()