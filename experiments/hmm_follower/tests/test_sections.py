"""sections.labels agrees with the three older labellings on every SAMBA take (reads only the annotation CSVs).

  cd experiments && yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_sections
"""
from __future__ import annotations

import glob
from pathlib import Path
import unittest

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[2]
SAMBA = EXPERIMENTS.parent / "Dataset" / "SAMBA dataset"


@unittest.skipUnless(SAMBA.exists(), "needs the SAMBA dataset")
class Sections(unittest.TestCase):
    def test_same_as_the_old_labellings(self):
        import importlib.util
        from hmm_follower import legacy  # noqa: F401
        from hmm_follower.sections import labels
        from melody_evidence import section_kinds
        from sweep_fixed_weights import piano_roles
        chord_rank = None  # harmony_cue/ is not in git yet: compare with it only where it exists
        if (EXPERIMENTS / "harmony_cue/chord_rank.py").exists():
            spec = importlib.util.spec_from_file_location("chord_rank_labels", EXPERIMENTS / "harmony_cue/chord_rank.py")
            chord_rank = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chord_rank)
        folders = sorted(glob.glob(str(SAMBA / "*_audio/songs/*/*/annotations/sections.csv")))
        self.assertEqual(len(folders), 152)
        for path in folders:
            parts = Path(path).parts
            performance_id = f"{parts[-6].split('_')[0]}/{parts[-4]}/{parts[-3]}"
            times = np.arange(-5., 900., .1)  # before, through and after every take
            new = labels(SAMBA, performance_id, times)
            with self.subTest(performance_id):
                if chord_rank is not None:
                    np.testing.assert_array_equal(new["kind"],
                                                  chord_rank.section_kinds(SAMBA, performance_id, times).astype(str))
                np.testing.assert_array_equal(new["coarse"], section_kinds(SAMBA, performance_id, times).astype(str))
                np.testing.assert_array_equal(new["piano"], piano_roles(SAMBA, performance_id, times).astype(str))


if __name__ == "__main__":
    unittest.main()
