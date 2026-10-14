"""Load the production directional-rule engine."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def rule_modules() -> dict:
    helpers = Path(__file__).resolve().parents[2] / "reproduction/backend/helpers"
    if str(helpers) not in sys.path:
        sys.path.insert(0, str(helpers))
    import generator_preparation as engine

    def canonical_side(side, isomeric=False):
        try:
            molecules = [engine._lenient_mol_from_smiles(part) for part in engine._side_parts(str(side))]
            return engine._canonical_side_from_mols(molecules, isomeric=isomeric)
        except Exception as exc:
            return {"ok": False, "smiles": None, "error": str(exc)}

    return {
        "_canonical_unmapped_side": canonical_side,
        **{name: getattr(engine, name) for name in (
            "_strip_rule_template_spectators", "_load_rule_template_prior_and_eval_records",
            "_resolve_rule_template_candidates", "_ec_level_dict_from_text",
            "_run_rule_smarts_on_reactant_side", "_rule_source_depth",
        )},
    }


def rule_paths(data_root: Path, output: Path) -> dict:
    source = data_root / "reaction_inputs/directional_rules"
    split = data_root / "reaction_inputs/reaction_split/split_assignments.parquet"
    if not split.is_file():
        raise FileNotFoundError(split)
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No directional-rule partitions: {source}")
    target = output / ".rule_inputs/reaction_labels"
    target.mkdir(parents=True, exist_ok=True)
    for path in files:
        destination = target / path.name
        if destination.exists():
            if not os.path.samefile(path, destination):
                if path.read_bytes() != destination.read_bytes():
                    raise FileExistsError(f"Rule input differs: {destination}")
            continue
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)
    return {"rule_chunks_dir": target.parent, "split_assignments_path": split}
