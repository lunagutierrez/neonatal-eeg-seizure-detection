import time
import pandas as pd
from config import WINDOWS, CHUNKS, FREQS, EXPERTS, INPUT_DIR, OUTPUT_DIR
from loaders import make_data_filename, load_hdf5_data
from processor import process_and_segment
from engine import run_kfold_experiment
from logger import get_logger
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger("MainPipeline")


def main():
    logger.info("Starting experiment pipeline")
    logger.info(f"Windows: {WINDOWS}")
    logger.info(f"Chunks: {CHUNKS}")
    logger.info(f"Experts: {EXPERTS}")

    for expert in EXPERTS:
        logger.info(f"=== Starting Expert: {expert} ===")

        # --- 1. Set per-expert input folder ---
        if expert == "A":
            expert_input_dir = INPUT_DIR / "results_expert_A"
        elif expert == "B":
            expert_input_dir = INPUT_DIR / "results_expert_B"
        elif expert == "C":
            expert_input_dir = INPUT_DIR / "results_expert_C"
        else:
            raise ValueError(f"Unknown expert: {expert}")

        # --- 2. Prepare expert output folder ---
        expert_out_dir = OUTPUT_DIR / expert
        expert_out_dir.mkdir(parents=True, exist_ok=True)

        # --- 3. Initialize results DataFrame ---
        results = pd.DataFrame(index=WINDOWS, columns=CHUNKS)

        for w in WINDOWS:
            for c in CHUNKS:
                start_time = time.time()
                logger.info(f"[START] Expert={expert} | W={w}s | C={c}")

                try:
                    # 1. Data Import
                    logger.info("Phase 1/3: Loading data")
                    # prepend the expert input folder to the filename
                    fname = expert_input_dir / make_data_filename(expert, w, c, FREQS)
                    raw_data = load_hdf5_data(fname)

                    # 2. Data Processing
                    logger.info("Phase 2/3: Processing & segmenting")
                    x, y = process_and_segment(raw_data, w * FREQS)
                    if x is None:
                        logger.warning("No windows generated, skipping")
                        continue

                    logger.info(f"Generated {x.shape[0]} windows")

                    # 3. Modeling
                    logger.info("Phase 3/3: K-fold training")
                    avg_acc = run_kfold_experiment(x, y)
                    results.at[w, c] = f"{avg_acc*100:.1f}%"

                    elapsed = time.time() - start_time
                    logger.info(
                        f"[DONE] Expert={expert} | W={w}s | C={c} | Acc={avg_acc:.2f} | Time={elapsed/60:.1f} min"
                    )

                    # Save intermediate results per expert
                    results_file = expert_out_dir / f"results_W{w}_C{c}.csv"
                    results.to_csv(results_file)

                except Exception:
                    logger.exception(f"[FAILED] Expert={expert} | W={w}s | C={c}")

        logger.info(f"=== Finished Expert: {expert} ===")
        logger.info("Experiment results:")
        print(results)


if __name__ == "__main__":
    main()
