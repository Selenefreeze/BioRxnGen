"""Audit elemental balance of existing candidate hits at their original ranks."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem

from benchmark import GROUPS, RESULTS
from ranking import load_ranked
from structures import side_keys


@lru_cache(maxsize=65536)
def composition(smiles):
    elements = Counter()
    charge = 0
    for part in smiles.split("."):
        mol = Chem.MolFromSmiles(part)
        if mol is None or mol.GetNumAtoms() == 0 or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return None
        elements.update(atom.GetAtomicNum() for atom in Chem.AddHs(mol).GetAtoms())
        charge += Chem.GetFormalCharge(mol)
    heavy = tuple(sorted((element, count) for element, count in elements.items() if element != 1))
    return heavy, elements[1], charge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_ranked(RESULTS / "ranked_candidates.zip")
    assays = pd.read_csv(RESULTS / "ranked_analytes.tsv", sep="\t").query("valid and enriched")
    targets = set(assays.structure_key)
    pools = defaultdict(dict)
    counts = Counter()
    fields = ["model_id", "label", "candidate_rank", "reactant_side_smiles", "product_full_nonisomeric",
              "generated_product_side_smiles"]
    for model, label, rank, left, right, product in rows[fields].itertuples(index=False, name=None):
        a, b = composition(left), composition(right)
        comparable = a is not None and b is not None
        balanced = {
            "all_candidates": True,
            "heavy_atoms_balanced": comparable and a[0] == b[0],
            "elements_and_charge_balanced": comparable and a == b,
        }
        counts["unparseable_sides"] += not comparable
        matched = targets.intersection(side_keys(product, label) - side_keys(left, label))
        for criterion, keep in balanced.items():
            if keep:
                counts[criterion] += 1
                pool = pools[criterion, model]
                for key in matched:
                    pool[key] = min(rank, pool.get(key, rank))
    groups = {row["cohort"]: row["models"] for row in GROUPS}
    summary = []
    for criterion in balanced:
        observed = assays.copy()
        observed["rank"] = [
            min((pools[criterion, m].get(key, float("inf")) for m in groups[cohort]), default=float("inf"))
            for cohort, key in observed[["cohort", "structure_key"]].itertuples(index=False, name=None)
        ]
        if criterion == "all_candidates":
            assert observed["rank"].replace(float("inf"), float("nan")).equals(observed.best_rank)
        for group, frame in [("combined", observed), *observed.groupby("study")]:
            summary.append({
                "criterion": criterion, "study": group, "observations": len(frame),
                "top5_hits": int(frame["rank"].le(5).sum()),
                "full_pool_hits": int(frame["rank"].lt(float("inf")).sum()),
            })
    result = pd.DataFrame(summary)
    reference = RESULTS / "reaction_balance_audit.tsv"
    if args.check:
        pd.testing.assert_frame_equal(result, pd.read_csv(reference, sep="\t"))
    if args.output:
        if args.output.resolve() == reference.resolve():
            raise ValueError("Use a separate output file")
        result.to_csv(args.output, sep="\t", index=False, lineterminator="\n")
    print(dict(counts))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
