from pathlib import Path
import json
CONFIG=json.loads((Path(__file__).resolve().parents[1]/'config/seed_extension.json').read_text())
TRAINING_SEED=int(CONFIG['training_seed'])
HEAD_SEED=int(CONFIG['head_seed'])
assert TRAINING_SEED == 3 and HEAD_SEED == TRAINING_SEED
