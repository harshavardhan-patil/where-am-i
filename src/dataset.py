from pathlib import Path

from tqdm import tqdm

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from huggingface_hub import snapshot_download

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR
    # ----------------------------------------------
):
# download the full dataset

    snapshot_download(repo_id="osv5m/osv5m", local_dir=RAW_DATA_DIR / "osv5m", repo_type='dataset')



if __name__ == "__main__":
    main()
