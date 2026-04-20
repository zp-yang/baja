from pathlib import Path
import os

workspace_dir = Path(os.path.abspath(__file__)).resolve().parent.parent.parent
# print(workspace_dir)
data_dir = workspace_dir / "data"
model_dir = workspace_dir / "models"
script_dir = workspace_dir / "scripts"

