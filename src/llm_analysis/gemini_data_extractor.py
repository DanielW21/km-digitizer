import io
import json
import os
import time

import matplotlib.pyplot as plt
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class KMDigitizer:
    def __init__(self, output_dir="logs"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash"
        self.output_base_dir = output_dir

    def _plot_to_image(self, data: dict) -> PIL.Image.Image:
        """Renders current data for visual comparison."""
        plt.figure(figsize=(8, 5))
        for group, points in data.items():
            if not points:
                continue
            pts = list(zip(*points))
            plt.step(pts[0], pts[1], where="post", label=group, linewidth=2)

        plt.ylim(0, 1.05)
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title("Reconstructed KM Plot")
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        buf.seek(0)
        return PIL.Image.open(buf)

    def save_iteration(self, iteration: int, data: dict, img: PIL.Image.Image, feedback: str):
        """Saves JSON and Image for the current attempt."""
        # Create a unique subfolder for this specific run if needed,
        # or just save to the output_base_dir
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)

        timestamp = int(time.time())
        base_filename = f"pass_{iteration}_{timestamp}"

        # Save JSON
        with open(os.path.join(self.output_base_dir, f"{base_filename}.json"), "w") as f:
            json.dump({"data": data, "feedback": feedback}, f, indent=2)

        # Save Image
        img.save(os.path.join(self.output_base_dir, f"{base_filename}.png"))

    def process_graph(
        self, image_path: str, previous_data: dict = None, feedback: str = ""
    ) -> dict:
        img = PIL.Image.open(image_path)
        mode = "Extract" if not previous_data else "Refine"

        prompt = f"""
            Act as a high-precision clinical data extractor.
            {mode} the data from this Kaplan-Meier plot.

            INSTRUCTIONS FOR HIGH FIDELITY:
            1. **Dynamic Sampling**: Extract a coordinate point for every single 'step' or 'drop' visible on the curves.
            2. **Grid Alignment**: Use the background grid lines and axis ticks as anchors to ensure X and Y values are mathematically consistent with the plot's scale.
            3. **Preserve Curvature**: For curves that appear smooth (high-N populations), ensure you capture enough points to reconstruct the slope without 'blocky' artifacts.
            4. **Handle Overlap**: If curves cross, trace the specific color/style to maintain the correct trajectory for each group.

            {f"FIX THIS OVER-SIMPLIFIED DATA: {json.dumps(previous_data)}" if previous_data else ""}
            {f"VALIDATOR CRITIQUE: {feedback}" if feedback else ""}

            Return ONLY a JSON object: {{"Group": [[x, y], ...]}}
            """

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)

    def validate_with_data(
        self, original_img_path: str, recon_img: PIL.Image.Image, current_data: dict
    ) -> tuple[bool, str]:
        original = PIL.Image.open(original_img_path)

        prompt = f"""
        Compare these two images and the raw data.
        IMAGE 1: Original Graph
        IMAGE 2: Reconstruction
        RAW DATA: {json.dumps(current_data)}

        Does Image 2 accurately represent the data shown in Image 1?
        Answer with a binary 'yes' or 'no' for the 'represents_data' field.

        Return JSON: {{"represents_data": bool, "feedback": "string explaining why if no"}}
        """

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, original, recon_img],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        res = json.loads(response.text)
        return res.get("represents_data", False), res.get("feedback", "")

    def run(self, image_path: str, max_attempts: int = 3) -> dict:
        data = {}
        feedback = ""

        for i in range(max_attempts):
            iteration = i + 1
            print(f"🔄 Pass {iteration}...")

            data = self.process_graph(
                image_path, previous_data=data if i > 0 else None, feedback=feedback
            )
            time.sleep(2)

            recon_img = self._plot_to_image(data)

            # This now exists!
            self.save_iteration(iteration, data, recon_img, feedback)

            is_valid, feedback = self.validate_with_data(image_path, recon_img, data)

            if is_valid:
                print(f"✅ Pass {iteration}: Validation Successful.")
                return data
            else:
                print(f"❌ Pass {iteration} Failed: {feedback}")
                time.sleep(2)

        print(f"⚠️ Max attempts ({max_attempts}) reached. Returning best-effort data.")
        return data


if __name__ == "__main__":
    digitizer = KMDigitizer()
    result_data = digitizer.run("sample_km_plot.png", max_attempts=4)  # use data folder plots
    print(json.dumps(result_data, indent=2))
    digitizer._plot_to_image(result_data).show()
