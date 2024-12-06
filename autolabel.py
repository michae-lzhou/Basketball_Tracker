from autodistill.detection import CaptionOntology
import torch
print(f'CUDA {torch.cuda.is_available()}')

ontology = CaptionOntology({
    "basketball orange ball": "Basketball",
})

IMG_DIR_PATH = "_raw_frames/tufts_v_brandeis"
DATA_DIR_PATH = "dataset"

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3

from autodistill_grounding_dino import GroundingDINO

base_model = GroundingDINO(ontology=ontology, box_threshold=BOX_THRESHOLD,
                           text_threshold=TEXT_THRESHOLD)

dataset = base_model.label(input_folder=IMG_DIR_PATH, extension=".jpg",
                           output_folder=DATA_DIR_PATH)


