# Run this once in a Python shell to fix the existing corrupted checkpoints
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from pathlib import Path

corrupted_dir = "/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/checkpoints/NNN_NC_SEED63_augm_0430_1748/hf_export"

# Load with ignore_mismatched_sizes — this reinits the 1-class head
model = RTDetrForObjectDetection.from_pretrained(
    corrupted_dir,
    ignore_mismatched_sizes=True
)
processor = RTDetrImageProcessor.from_pretrained(corrupted_dir)

# Patch config
model.config.id2label  = {0: "dugong"}
model.config.label2id  = {"dugong": 0}
model.config.num_labels = 1

# Overwrite with clean weights
model.save_pretrained(corrupted_dir)
processor.save_pretrained(corrupted_dir)
print("Fixed!")