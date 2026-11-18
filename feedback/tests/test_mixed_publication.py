import copy
import unittest
import numpy as np
from drivesta_feedback.batch.scene_store import ROW_DTYPE, validate_rows
from drivesta_feedback.batch.mixed_provenance import reconstruct, record_digest


class MixedPublicationTest(unittest.TestCase):
    def setUp(self):
        seed = np.zeros((8, 3), np.float32)
        seed[:, 0] = np.arange(1, 9) * 2
        self.group = dict(
            operator="interior",
            seed=seed.tolist(),
            offset_keys=[[0, 0], [100, 20]],
            record_digests={},
        )
        self.rows = np.zeros(2, dtype=ROW_DTYPE)
        self.rows["keys"] = [[0, 0], [0, 1]]
        self.rows["trajectories"] = reconstruct(self.group, np.arange(2))
        self.rows["labels"] = 1
        for i, row in enumerate(self.rows):
            self.group["record_digests"][str(i)] = record_digest(row["trajectories"], row["labels"])
        self.protocol = dict(
            constructor=dict(
                format="verified_mixed_recovery_v1",
                key_semantics="construction_group_and_query_index",
                groups=[self.group],
            )
        )

    def test_full_and_subset(self):
        validate_rows(self.rows, None, self.protocol)
        validate_rows(self.rows[1:], None, self.protocol)

    def test_heading_corruption(self):
        self.rows["trajectories"][1, 4, 2] += 0.01
        with self.assertRaises(ValueError):
            validate_rows(self.rows, None, self.protocol)

    def test_self_consistent_label_corruption(self):
        self.rows["labels"][0, 2] = 0.8
        self.rows["labels"][0, 6] = (5 * 0.8 + 7) / 12
        with self.assertRaises(ValueError):
            validate_rows(self.rows, None, self.protocol)

    def test_coefficients_corruption(self):
        self.group["offset_keys"][1][0] += 50
        with self.assertRaises(ValueError):
            validate_rows(self.rows, None, self.protocol)

    def test_unknown_group_or_query(self):
        for key in ((1, 0), (0, 2), (-1, 0)):
            rows = self.rows.copy()
            rows["keys"][0] = key
            with self.assertRaises(ValueError):
                validate_rows(rows, None, self.protocol)

    def test_unknown_operator(self):
        self.group["operator"] = "unchecked"
        with self.assertRaises(ValueError):
            validate_rows(self.rows, None, self.protocol)

    def test_legacy_quintic_gate_still_applies(self):
        protocol = dict(
            constructor={"max_planned_endpoint_error_m": 3e-5},
            sampling={"integer_lattice_m": 0.001},
        )
        with self.assertRaises(ValueError):
            validate_rows(self.rows, np.array(self.group["seed"]), protocol)


if __name__ == "__main__":
    unittest.main()
