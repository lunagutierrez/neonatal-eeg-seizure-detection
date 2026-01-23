import time
import pandas as pd
from config import WINDOWS, CHUNKS, FREQS
from loaders import make_data_filename, load_hdf5_data
from processor import process_and_segment
from engine import run_kfold_experiment
from logger import get_logger

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger("MainPipeline")

def main():
    logger.info("Starting experiment pipeline")
    logger.info("Expert: B")
    logger.info(f"Windows: {WINDOWS}")
    logger.info(f"Chunks: {CHUNKS}")

    results = pd.DataFrame(index=WINDOWS, columns=CHUNKS)

    for w in WINDOWS:
        for c in CHUNKS:
            start_time = time.time()
            logger.info(f"[START] W={w}s | C={c}")

            try:
                # 1. Data Import
                logger.info("Phase 1/3: Loading data")
                fname = make_data_filename("B", w, c, FREQS)
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
                    f"[DONE] W={w}s | C={c} | Acc={avg_acc:.2f} | Time={elapsed/60:.1f} min"
                )

            except Exception:
                logger.exception(f"[FAILED] W={w}s | C={c}")

    logger.info("Experiment results")
    print(results)

if __name__ == "__main__":
    main()