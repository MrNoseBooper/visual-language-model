"""CLIP's picture encoder: turns a picture into 512 numbers (an "embedding").

Only CLIP's picture half is loaded. It knows NO item names: it just describes what a picture looks
like as numbers, so similar-looking pictures get similar numbers. The names come only from your own
photo folders (data/<label>/), when train.py teaches a small model which numbers belong to which item.

CLIP is downloaded from Hugging Face the first time (one ~600 MB file; only its picture half is loaded)
and kept in models/clip/.
Licence: MIT.
"""
import cv2
import numpy as np

from . import FOLDER

MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_DIR = FOLDER / "models" / "clip"
BATCH_SIZE = 32               # pictures processed at once during training


class Clip:
    def __init__(self):
        print(f"Loading CLIP's picture encoder ({MODEL_NAME}) ...")
        # Imported here, not at the top, so collect.py never needs PyTorch.
        import torch
        from transformers import CLIPVisionModelWithProjection
        from transformers import logging as transformers_logging
        try:   # newer Transformers: the picture preparer that doesn't need torchvision
            from transformers import CLIPImageProcessorPil as ImageProcessor
        except ImportError:
            from transformers import CLIPImageProcessor as ImageProcessor

        # Hide the notice that CLIP's text half isn't being used (that's on purpose).
        transformers_logging.set_verbosity_error()
        self.torch = torch
        # Use only the copy on this laptop once it's downloaded: no internet check on every start.
        offline = CACHE_DIR.exists() and any(CACHE_DIR.glob("models--*"))
        # Load once, never per frame. eval() switches off training-only behaviour.
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=offline).eval()
        self.processor = ImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=offline)
        print("CLIP loaded" + (" (from this laptop, no internet used)." if offline else "."))

    def embed_images(self, images):
        """512 numbers for each picture (OpenCV BGR images), scaled to length 1. Shape (n, 512)."""
        rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        chunks = []
        for start in range(0, len(rgb), BATCH_SIZE):
            inputs = self.processor(images=rgb[start:start + BATCH_SIZE], return_tensors="pt")
            with self.torch.no_grad():   # no training here, so skip the extra bookkeeping
                chunks.append(self.model(pixel_values=inputs["pixel_values"]).image_embeds.numpy())
        embeddings = np.concatenate(chunks)
        # Scale each row to length 1, so all pictures are compared on the same scale.
        return embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
