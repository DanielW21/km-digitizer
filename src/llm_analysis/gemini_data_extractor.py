import io
import json
import os
import re
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scipy.interpolate import PchipInterpolator

load_dotenv()


class KMDigitizer:
    def __init__(self, output_dir="logs"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash"

    def _ensure_monotonic(self, points: list) -> list:
        if not points:
            return []

        sorted_pts = sorted(points, key=lambda x: x[0])
        monotonic_pts = [sorted_pts[0]]

        count = 0
        for i in range(1, len(sorted_pts)):
            prev_y = monotonic_pts[-1][1]
            curr_x, curr_y = sorted_pts[i]

            if curr_y > prev_y:
                curr_y = prev_y
                count += 1

            monotonic_pts.append([curr_x, curr_y])

        if count > 0:
            print(f"  - Adjusted {count} points to enforce monotonicity.")

        return monotonic_pts

    def _safe_json_parse(self, text: str) -> dict:
        try:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            json_str = match.group(1) if match else text
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            return {}

    def _apply_smoothing(self, data: dict, smoothing_config: dict) -> dict:
        smoothed_data = {}
        for group, points in data.items():
            if not points or len(points) < 3:
                smoothed_data[group] = self._ensure_monotonic(points)
                continue

            strength = min(smoothing_config.get(group, 0), 2)
            if strength < 1:
                smoothed_data[group] = self._ensure_monotonic(points)
                continue

            sorted_pts = self._ensure_monotonic(points)
            x = np.array([p[0] for p in sorted_pts])
            y = np.array([p[1] for p in sorted_pts])

            x_unique, unique_indices = np.unique(x, return_index=True)
            y_unique = y[unique_indices]

            if len(x_unique) < 3:
                smoothed_data[group] = sorted_pts
                continue

            try:
                interp_func = PchipInterpolator(x_unique, y_unique)

                # density approach capped at a conservative level, n + (n-1) * strength for 0-2 strength
                new_x = np.linspace(
                    x_unique.min(), x_unique.max(), len(x_unique) + (len(x_unique) - 1) * strength
                )
                new_y = interp_func(new_x)

                interp_points = [[float(cx), float(cy)] for cx, cy in zip(new_x, new_y)]
                smoothed_data[group] = self._ensure_monotonic(interp_points)
            except Exception:
                smoothed_data[group] = sorted_pts
        return smoothed_data

    def _plot_to_image(self, data: dict) -> PIL.Image.Image:
        plt.figure(figsize=(8, 5))
        has_plotted = False
        for group, points in data.items():
            if not points:
                continue
            try:
                sorted_pts = sorted(points, key=lambda x: x[0])
                pts = list(zip(*sorted_pts))

                plt.step(pts[0], pts[1], where="post", label=group, linewidth=2)
                has_plotted = True
            except Exception:
                pass

        if not has_plotted:
            plt.text(0.5, 0.5, "No valid data extracted", ha="center", va="center")

        plt.ylim(-0.02, 1.05)
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        buf.seek(0)
        return PIL.Image.open(buf)

    def process_graph(
        self, image_path: str, previous_data: dict = None, feedback: str = ""
    ) -> dict:
        try:
            img = PIL.Image.open(image_path)
            mode = "Extract" if not previous_data else "Refine"
            prompt = f"""
            Act as a high-precision clinical data extractor. {mode} the Kaplan-Meier data.

            INSTRUCTIONS FOR HIGH FIDELITY:
            1. **Dynamic Sampling**: Extract a coordinate point for every single 'step' or 'drop' visible on the curves.
            2. **Grid Alignment**: Use the background grid lines and axis ticks as anchors to ensure X and Y values are mathematically consistent with the plot's scale.
            3. **Preserve Curvature**: For curves that appear smooth (high-N populations), ensure you capture enough points to reconstruct the slope without 'blocky' artifacts.
            4. **Handle Overlap**: If curves cross, trace the specific color/style to maintain the correct trajectory for each group.

            {f"PREVIOUS DATA: {json.dumps(previous_data)}" if previous_data else ""}
            {f"VALIDATOR CRITIQUE: {feedback}" if feedback else ""}
            Return ONLY JSON: {{"GroupName": [[x, y], ...]}}
            """
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, img],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            raw_data = self._safe_json_parse(response.text)
            processed_data = {k: self._ensure_monotonic(v) for k, v in raw_data.items()}
            return processed_data
        except Exception:
            return previous_data or {}

    def validate_with_data(
        self, original_img_path: str, recon_img: PIL.Image.Image, current_data: dict
    ) -> tuple[bool, str, dict]:
        try:
            original = PIL.Image.open(original_img_path)
            prompt = f"""
            Compare Original (Image 1) and Reconstruction (Image 2).
            DATA: {json.dumps(current_data)}
            1. Does Image 2 accurately represent Image 1? If yes, return 'represents_data': true.
            2. Smoothing: Assign a 'smoothing_strength' (0-2) to the GroupName if Image 1 is a smooth decline but Image 2 is jagged steps.
               - 1: Minor interpolation (1 point between steps)
               - 2: Moderate interpolation (2 points between steps)
               - 0: Keep same number of steps (recommended for standard KM plots)
            Return ONLY JSON: {{"represents_data": bool, "feedback": "string", "smoothing": {{"GroupName": int}}}}
            """
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, original, recon_img],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            res = self._safe_json_parse(response.text)
            return (
                res.get("represents_data", False),
                res.get("feedback", ""),
                res.get("smoothing", {}),
            )
        except Exception:
            return False, "Validation Error", {}

    def save_iteration(
        self, target_dir: str, iteration: int, data: dict, img: PIL.Image.Image, feedback: str
    ):
        os.makedirs(target_dir, exist_ok=True)
        base = f"pass_{iteration}"
        with open(os.path.join(target_dir, f"{base}.json"), "w") as f:
            json.dump({"data": data, "feedback": feedback}, f, indent=2)
        img.save(os.path.join(target_dir, f"{base}.png"))

    def run(self, image_path: str, target_dir: str, max_attempts: int = 3) -> dict:
        data = {}
        feedback = ""
        for i in range(max_attempts):
            iteration = i + 1
            print(f"Starting Pass {iteration}...")
            try:
                new_data = self.process_graph(
                    image_path, previous_data=data or None, feedback=feedback
                )
                if not new_data and not data:
                    continue
                data = new_data

                time.sleep(30)

                recon_raw = self._plot_to_image(data)
                is_valid, feedback, smooth_cfg = self.validate_with_data(
                    image_path, recon_raw, data
                )

                if smooth_cfg:
                    print(f"Pass {iteration}: Applying conservative smoothing (Strength 0-2)")
                    data = self._apply_smoothing(data, smooth_cfg)
                    recon_final = self._plot_to_image(data)
                else:
                    recon_final = recon_raw

                self.save_iteration(target_dir, iteration, data, recon_final, feedback)

                if is_valid:
                    print(f"Pass {iteration}: Validation successful")
                    return data
                else:
                    print(f"Pass {iteration}: Validation failed. Reason: {feedback[:100]}")

            except Exception as e:
                print(f"Pass {iteration}: Internal error encountered")
                traceback.print_exc()

            if i < max_attempts - 1:
                time.sleep(30)

        print("Max attempts reached")
        return data


if __name__ == "__main__":
    digitizer = KMDigitizer()
    test_image = "sample_km_plot.png"
    if os.path.exists(test_image):
        result = digitizer.run(test_image, target_dir="logs/debug_run")
        print("Process complete")
