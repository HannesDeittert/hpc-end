from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple


def mkdir_recursive(path: str) -> None:
    subfolders = []
    while not os.path.isdir(path):
        path, subfolder = os.path.split(path)
        subfolders.append(subfolder)

    for subfolder in reversed(subfolders):
        path = os.path.join(path, subfolder)
        if not os.path.isdir(path):
            os.mkdir(path)


def get_result_checkpoint_config_and_log_path(
    *, all_results_folder: str, name: str
) -> Tuple[str, str, str, str]:
    """Create result CSV + run folder + checkpoint folder + main log path.

    Copied from `stEVE_training/training_scripts/util/util.py` so our
    training entrypoints don't depend on that repo.
    """

    today = datetime.today().strftime("%Y-%m-%d")
    time = datetime.today().strftime("%H%M%S")

    main_resultfile = os.path.join(all_results_folder, f"{today}_{time}_{name}.csv")

    file_id = 0
    while os.path.isfile(main_resultfile):
        file_id += 1
        main_resultfile = os.path.join(
            all_results_folder, f"{today}_{time}_{name}_{file_id}.csv"
        )

    results_folder = main_resultfile[:-4]
    checkpoint_folder = os.path.join(results_folder, "checkpoints")
    mkdir_recursive(checkpoint_folder)

    log_file = os.path.join(results_folder, "main.log")
    return main_resultfile, checkpoint_folder, results_folder, log_file
