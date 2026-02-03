import json
import os
import shutil
from datetime import datetime

from tqdm import tqdm

from llm_analysis.gemini_data_extractor import KMDigitizer

SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def run_digitization_task(file_path):
    session_dir = os.path.join("logs", SESSION_TIMESTAMP)

    # Sub Folder Ex: logs/20260202_1845/plot_0000_standard/
    input_name = os.path.splitext(os.path.basename(file_path))[0]
    plot_dir = os.path.join(session_dir, input_name)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" PROCESSING: {input_name}")
    print(f"{'='*60}")

    digitizer = KMDigitizer()

    final_data = digitizer.run(file_path, target_dir=plot_dir, max_attempts=5)

    json_path = os.path.join(plot_dir, "final_extracted_data.json")
    with open(json_path, "w") as f:
        json.dump(final_data, f, indent=2)

    recon_img = digitizer._plot_to_image(final_data)
    recon_img.save(os.path.join(plot_dir, "final_reconstructed_plot.png"))

    shutil.copy2(file_path, os.path.join(plot_dir, "original_input.png"))

    print(f"Task Finished. Files saved in {plot_dir}")
    return final_data


if __name__ == "__main__":
    dataset_configs = [
        {"name": "Synthetic Benchmark", "dir": "data/synthetic_data/benchmark/plots/"},
        {"name": "Real Data", "dir": "data/real_data/"},
    ]

    for config in dataset_configs:
        path = config["dir"]
        if os.path.exists(path):
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            print(f"\n Starting {config['name']} Batch...")

            for file_path in tqdm(files, desc=f"Digitizing {config['name']}", unit="plot"):
                try:
                    run_digitization_task(file_path)
                except Exception as e:
                    print(f"\nError processing {file_path}: {e}")
        else:
            print(f"Directory not found: {path}")

    print(f"\nAll tasks complete. Logs in: logs/{SESSION_TIMESTAMP}")
