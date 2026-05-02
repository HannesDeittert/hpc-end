from __future__ import annotations

import unittest

import numpy as np

from steve_recommender.evaluation.info_collectors import SofaWallForceInfo


class WallForceMappingTests(unittest.TestCase):
    def test_resolve_constraint_dt_prefers_simulation_root_dt(self) -> None:
        info = SofaWallForceInfo(
            object(),
            mode="constraint_projected_si_validated",
            constraint_dt_s=0.2,
        )

        class _Data:
            def __init__(self, value):
                self.value = value

        class _Root:
            dt = _Data(0.006)

        class _Sim:
            root = _Root()

        dt = info._resolve_constraint_dt_s(_Sim())
        self.assertIsNotNone(dt)
        self.assertAlmostEqual(float(dt), 0.006, places=9)

    def test_extract_tip_force_from_samples_uses_nearest_active_sample(self) -> None:
        vec, norm, idx, source = SofaWallForceInfo._extract_tip_force_from_samples(
            tip_pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            candidate_forces=np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32
            ),
            candidate_positions=np.asarray(
                [[5.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32
            ),
            contact_epsilon=1e-7,
        )
        self.assertTrue(np.allclose(vec, np.asarray([0.0, 2.0, 0.0], dtype=np.float32)))
        self.assertAlmostEqual(norm, 2.0, places=6)
        self.assertEqual(idx, 1)
        self.assertEqual(source, "nearest_tip_sample")

    def test_extract_tip_force_reports_below_epsilon(self) -> None:
        vec, norm, idx, source = SofaWallForceInfo._extract_tip_force_from_samples(
            tip_pos=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            candidate_forces=np.asarray([[1e-9, 0.0, 0.0]], dtype=np.float32),
            candidate_positions=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            contact_epsilon=1e-7,
        )
        self.assertTrue(np.allclose(vec, np.zeros((3,), dtype=np.float32)))
        self.assertAlmostEqual(norm, 0.0, places=9)
        self.assertEqual(idx, 0)
        self.assertEqual(source, "nearest_tip_sample_below_epsilon")

    def test_maps_contact_points_to_nearest_wall_triangles(self) -> None:
        wall_centroids = np.asarray(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32
        )
        contact_pairs = [
            (
                np.asarray([1.1, 0.0, 0.0], dtype=np.float32),
                np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
            ),
            (
                np.asarray([8.9, 0.0, 0.0], dtype=np.float32),
                np.asarray([9.9, 0.0, 0.0], dtype=np.float32),
            ),
        ]
        candidate_positions = np.asarray(
            [[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32
        )
        candidate_forces = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32
        )

        dense, ids, active, total = SofaWallForceInfo._map_contact_forces_to_wall_segments(
            contact_pairs=contact_pairs,
            wall_centroids=wall_centroids,
            candidate_forces=candidate_forces,
            candidate_positions=candidate_positions,
            contact_epsilon=1e-7,
        )

        self.assertEqual(dense.shape, (2, 3))
        self.assertTrue(np.allclose(dense[0], np.asarray([-1.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(dense[1], np.asarray([0.0, -2.0, 0.0])))
        self.assertTrue(np.array_equal(ids, np.asarray([0, 1], dtype=np.int32)))
        self.assertEqual(active.shape, (2, 3))
        self.assertTrue(np.allclose(total, np.asarray([-1.0, -2.0, 0.0], dtype=np.float32)))

    def test_skips_small_forces_below_epsilon(self) -> None:
        wall_centroids = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        contact_pairs = [
            (
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            )
        ]
        candidate_positions = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        candidate_forces = np.asarray([[1e-9, 0.0, 0.0]], dtype=np.float32)

        dense, ids, active, total = SofaWallForceInfo._map_contact_forces_to_wall_segments(
            contact_pairs=contact_pairs,
            wall_centroids=wall_centroids,
            candidate_forces=candidate_forces,
            candidate_positions=candidate_positions,
            contact_epsilon=1e-7,
        )

        self.assertTrue(np.allclose(dense, np.zeros((1, 3), dtype=np.float32)))
        self.assertEqual(ids.size, 0)
        self.assertEqual(active.size, 0)
        self.assertTrue(np.allclose(total, np.zeros((3,), dtype=np.float32)))

    def test_contact_records_use_explicit_triangle_id(self) -> None:
        wall_centroids = np.asarray(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32
        )
        records = [
            {
                "wire_point": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "wall_point": np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
                "wall_triangle_id": 0,
                "triangle_source": "contact_element_preferred_side",
            }
        ]
        candidate_positions = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        candidate_forces = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)

        (
            dense,
            ids,
            active,
            total,
            method,
            mapped,
            explicit_ratio,
            _mapped_force_count,
            _mapped_force_explicit_count,
        ) = SofaWallForceInfo._map_contact_records_to_wall_segments(
            contact_records=records,
            wall_vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [11.0, 0.0, 0.0],
                    [10.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            wall_triangles=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
            wall_centroids=wall_centroids,
            candidate_forces=candidate_forces,
            candidate_positions=candidate_positions,
            contact_epsilon=1e-7,
        )

        self.assertEqual(method, "contact_element_triangle_id")
        self.assertEqual(mapped, 1)
        self.assertAlmostEqual(explicit_ratio, 1.0, places=6)
        self.assertTrue(np.array_equal(ids, np.asarray([0], dtype=np.int32)))
        self.assertTrue(np.allclose(active[0], np.asarray([0.0, -1.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(dense[0], np.asarray([0.0, -1.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(total, np.asarray([0.0, -1.0, 0.0], dtype=np.float32)))

    def test_contact_records_use_native_explicit_triangle_id_method(self) -> None:
        wall_centroids = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        records = [
            {
                "wire_point": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "wall_point": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "wall_triangle_id": 0,
                "triangle_source": "native_contact_export_triangle_id",
            }
        ]
        candidate_positions = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        candidate_forces = np.asarray([[0.0, 2.0, 0.0]], dtype=np.float32)
        (
            dense,
            ids,
            active,
            total,
            method,
            mapped,
            explicit_ratio,
            mapped_force_count,
            mapped_force_explicit_count,
        ) = SofaWallForceInfo._map_contact_records_to_wall_segments(
            contact_records=records,
            wall_vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            wall_triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
            wall_centroids=wall_centroids,
            candidate_forces=candidate_forces,
            candidate_positions=candidate_positions,
            contact_epsilon=1e-7,
        )
        self.assertEqual(method, "native_contact_export_triangle_id")
        self.assertEqual(mapped, 1)
        self.assertEqual(mapped_force_count, 1)
        self.assertEqual(mapped_force_explicit_count, 1)
        self.assertAlmostEqual(explicit_ratio, 1.0, places=6)
        self.assertTrue(np.array_equal(ids, np.asarray([0], dtype=np.int32)))
        self.assertTrue(np.allclose(active[0], np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(dense[0], np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(total, np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))

    def test_listener_triangle_id_parser_prefers_wall_side(self) -> None:
        info = SofaWallForceInfo(object(), mode="constraint_projected_si_validated")

        class _Listener:
            # model2 is wall => parser should use second side.
            collisionModel1 = "InstrumentCombined/CollisionModel/PointCollisionModel"
            collisionModel2 = "vesselTree/TriangleCollisionModel"

            @staticmethod
            def getContactElements():
                return [
                    ((11, 0), (3, 0)),
                    ((12, 0), {"triangleIndex": 9}),
                ]

        ids = info._read_listener_wall_triangle_ids(_Listener(), wall_triangle_count=20)
        self.assertEqual(ids[0][0], 3)
        self.assertTrue(ids[0][1].startswith("contact_element"))
        self.assertEqual(ids[1][0], 9)
        self.assertTrue(ids[1][1].startswith("contact_element"))

    def test_contact_records_surface_mapping_without_explicit_triangle(self) -> None:
        wall_vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        wall_triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        wall_centroids = wall_vertices[wall_triangles].mean(axis=1)
        records = [
            {
                "wire_point": None,
            "wall_point": np.asarray([0.1, 0.1, 0.0], dtype=np.float32),
            "wall_triangle_id": None,
            "triangle_source": "contact_node_wall_point",
        }
        ]
        candidate_positions = np.asarray([[0.1, 0.1, 0.0]], dtype=np.float32)
        candidate_forces = np.asarray([[0.0, 2.0, 0.0]], dtype=np.float32)

        (
            dense,
            ids,
            active,
            total,
            method,
            mapped,
            explicit_ratio,
            _mapped_force_count,
            _mapped_force_explicit_count,
        ) = SofaWallForceInfo._map_contact_records_to_wall_segments(
            contact_records=records,
            wall_vertices=wall_vertices,
            wall_triangles=wall_triangles,
            wall_centroids=wall_centroids,
            candidate_forces=candidate_forces,
            candidate_positions=candidate_positions,
            contact_epsilon=1e-7,
        )

        self.assertEqual(method, "contact_point_nearest_triangle_surface")
        self.assertEqual(mapped, 1)
        self.assertAlmostEqual(explicit_ratio, 0.0, places=6)
        self.assertTrue(np.array_equal(ids, np.asarray([0], dtype=np.int32)))
        self.assertTrue(np.allclose(active[0], np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(dense[0], np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(total, np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))

    def test_constraint_projection_applies_dt_scaling(self) -> None:
        info = SofaWallForceInfo(object(), mode="constraint_projected_si_validated")
        projected, row_contribs = info._project_constraint_forces(
            lcp_forces=np.asarray([2.0], dtype=np.float32),
            constraint_raw="0 1 0 1 0 0",
            n_points=1,
            dt_s=0.5,
        )
        self.assertEqual(projected.shape, (1, 3))
        # coeff=[1,0,0], lambda=2, dt=0.5 -> force=4 along x
        self.assertTrue(np.allclose(projected[0], np.asarray([4.0, 0.0, 0.0], dtype=np.float32)))
        self.assertEqual(len(row_contribs), 1)
        self.assertEqual(int(row_contribs[0]["row_idx"]), 0)
        self.assertEqual(int(row_contribs[0]["dof_idx"]), 0)
        self.assertTrue(
            np.allclose(
                np.asarray(row_contribs[0]["force_vec"], dtype=np.float32),
                np.asarray([4.0, 0.0, 0.0], dtype=np.float32),
            )
        )

    def test_row_projected_mapping_reaches_explicit_coverage(self) -> None:
        wall_vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        wall_triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
        wall_centroids = wall_vertices[wall_triangles].mean(axis=1)
        row_contribs = [
            {
                "row_idx": 5,
                "dof_idx": 10,
                "force_vec": np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
                "force_norm": 2.0,
            }
        ]
        records = [
            {
                "wall_point": np.asarray([0.2, 0.2, 0.0], dtype=np.float32),
                "wall_triangle_id": 0,
                "triangle_source": "native_contact_export_triangle_id",
                "constraint_row_index": 5,
                "collision_dof_index": 10,
                "integrity_ordering_stable": True,
                "integrity_mapping_complete": True,
                "contact_kind": "line",
            }
        ]
        (
            dense,
            ids,
            active,
            total,
            method,
            mapped,
            explicit_ratio,
            mapped_force_count,
            mapped_force_explicit_count,
            gap_info,
        ) = SofaWallForceInfo._map_row_projected_forces_to_wall_segments(
            row_contribs=row_contribs,
            contact_records=records,
            wall_vertices=wall_vertices,
            wall_triangles=wall_triangles,
            wall_centroids=wall_centroids,
            contact_epsilon=1e-7,
            allow_surface_fallback=False,
        )
        self.assertEqual(method, "native_contact_export_triangle_id")
        self.assertEqual(mapped, 1)
        self.assertAlmostEqual(explicit_ratio, 1.0, places=6)
        self.assertEqual(mapped_force_count, 1)
        self.assertEqual(mapped_force_explicit_count, 1)
        self.assertTrue(np.array_equal(ids, np.asarray([0], dtype=np.int32)))
        self.assertTrue(np.allclose(active[0], np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(dense[0], np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(total, np.asarray([0.0, -2.0, 0.0], dtype=np.float32)))
        self.assertEqual(int(gap_info["active_projected_count"]), 1)
        self.assertEqual(int(gap_info["explicit_mapped_count"]), 1)
        self.assertEqual(int(gap_info["unmapped_count"]), 0)

    def test_row_projected_mapping_classifies_domain_mismatch(self) -> None:
        wall_vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        wall_triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
        wall_centroids = wall_vertices[wall_triangles].mean(axis=1)
        row_contribs = [
            {
                "row_idx": 7,
                "dof_idx": 42,
                "force_vec": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "force_norm": 1.0,
            }
        ]
        records = [
            {
                "wall_point": np.asarray([0.2, 0.2, 0.0], dtype=np.float32),
                "wall_triangle_id": 0,
                "triangle_source": "native_contact_export_triangle_id",
                "constraint_row_index": 7,
                "collision_dof_index": 5,  # mismatch to projected dof_idx=42
                "integrity_ordering_stable": True,
                "integrity_mapping_complete": True,
                "contact_kind": "point",
            }
        ]
        (
            _dense,
            _ids,
            _active,
            _total,
            method,
            mapped,
            explicit_ratio,
            mapped_force_count,
            mapped_force_explicit_count,
            gap_info,
        ) = SofaWallForceInfo._map_row_projected_forces_to_wall_segments(
            row_contribs=row_contribs,
            contact_records=records,
            wall_vertices=wall_vertices,
            wall_triangles=wall_triangles,
            wall_centroids=wall_centroids,
            contact_epsilon=1e-7,
            allow_surface_fallback=False,
        )
        self.assertEqual(method, "none")
        self.assertEqual(mapped, 0)
        self.assertAlmostEqual(explicit_ratio, 0.0, places=6)
        self.assertEqual(mapped_force_count, 0)
        self.assertEqual(mapped_force_explicit_count, 0)
        self.assertEqual(int(gap_info["active_projected_count"]), 1)
        self.assertEqual(int(gap_info["explicit_mapped_count"]), 0)
        self.assertEqual(int(gap_info["unmapped_count"]), 1)
        self.assertEqual(
            int(gap_info["class_counts"].get("domain_mismatch_row_vs_force_index", 0)),
            1,
        )

    def test_row_projected_mapping_bridges_local_to_global_rows(self) -> None:
        wall_vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        wall_triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
        wall_centroids = wall_vertices[wall_triangles].mean(axis=1)
        row_contribs = [
            {
                "row_idx": 20,  # global projection row
                "dof_idx": 10,
                "force_vec": np.asarray([0.0, 1.5, 0.0], dtype=np.float32),
                "force_norm": 1.5,
            }
        ]
        records = [
            {
                "wall_point": np.asarray([0.25, 0.25, 0.0], dtype=np.float32),
                "wall_triangle_id": 0,
                "triangle_source": "native_contact_export_triangle_id",
                "constraint_row_index": 2,  # local contact-node row (needs +18 bridge)
                "collision_dof_index": 10,
                "integrity_ordering_stable": True,
                "integrity_mapping_complete": True,
                "contact_kind": "line",
            }
        ]
        (
            dense,
            ids,
            active,
            total,
            method,
            mapped,
            explicit_ratio,
            mapped_force_count,
            mapped_force_explicit_count,
            gap_info,
        ) = SofaWallForceInfo._map_row_projected_forces_to_wall_segments(
            row_contribs=row_contribs,
            contact_records=records,
            wall_vertices=wall_vertices,
            wall_triangles=wall_triangles,
            wall_centroids=wall_centroids,
            contact_epsilon=1e-7,
            allow_surface_fallback=False,
        )
        self.assertEqual(method, "native_contact_export_triangle_id_global_row_bridge")
        self.assertEqual(mapped, 1)
        self.assertAlmostEqual(explicit_ratio, 1.0, places=6)
        self.assertEqual(mapped_force_count, 1)
        self.assertEqual(mapped_force_explicit_count, 1)
        self.assertTrue(np.array_equal(ids, np.asarray([0], dtype=np.int32)))
        self.assertTrue(np.allclose(active[0], np.asarray([0.0, -1.5, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(dense[0], np.asarray([0.0, -1.5, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(total, np.asarray([0.0, -1.5, 0.0], dtype=np.float32)))
        self.assertTrue(bool(gap_info.get("row_bridge_applied", False)))
        self.assertEqual(int(gap_info.get("row_bridge_offset", 0)), 18)

    def test_row_projected_mapping_filters_non_contact_dofs_from_coverage(self) -> None:
        wall_vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        wall_triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
        wall_centroids = wall_vertices[wall_triangles].mean(axis=1)
        row_contribs = [
            {
                "row_idx": 20,
                "dof_idx": 10,
                "force_vec": np.asarray([0.0, 1.5, 0.0], dtype=np.float32),
                "force_norm": 1.5,
            },
            {
                "row_idx": 30,  # unrelated non-contact dof contribution
                "dof_idx": 99,
                "force_vec": np.asarray([0.0, 0.5, 0.0], dtype=np.float32),
                "force_norm": 0.5,
            },
        ]
        records = [
            {
                "wall_point": np.asarray([0.25, 0.25, 0.0], dtype=np.float32),
                "wall_triangle_id": 0,
                "triangle_source": "native_contact_export_triangle_id",
                "constraint_row_index": 2,
                "collision_dof_index": 10,
                "integrity_ordering_stable": True,
                "integrity_mapping_complete": True,
                "contact_kind": "line",
            }
        ]
        (
            _dense,
            _ids,
            _active,
            _total,
            method,
            mapped,
            explicit_ratio,
            _mapped_force_count,
            _mapped_force_explicit_count,
            gap_info,
        ) = SofaWallForceInfo._map_row_projected_forces_to_wall_segments(
            row_contribs=row_contribs,
            contact_records=records,
            wall_vertices=wall_vertices,
            wall_triangles=wall_triangles,
            wall_centroids=wall_centroids,
            contact_epsilon=1e-7,
            allow_surface_fallback=False,
        )
        self.assertEqual(method, "native_contact_export_triangle_id_global_row_bridge")
        self.assertEqual(mapped, 1)
        self.assertAlmostEqual(explicit_ratio, 1.0, places=6)
        self.assertEqual(int(gap_info["active_projected_count"]), 1)
        self.assertEqual(int(gap_info["explicit_mapped_count"]), 1)
        self.assertEqual(int(gap_info["unmapped_count"]), 0)

    def test_contact_record_prefers_nonzero_force_sample(self) -> None:
        wall_vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        wall_triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
        wall_centroids = wall_vertices[wall_triangles].mean(axis=1)
        records = [
            {
                "wire_point": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "wall_point": np.asarray([0.1, 0.1, 0.0], dtype=np.float32),
                "wall_triangle_id": 0,
                "triangle_source": "contact_element_preferred_side",
            }
        ]
        candidate_positions = np.asarray(
            [
                [0.0, 0.0, 0.0],   # closest but zero force
                [0.2, 0.0, 0.0],   # slightly farther, non-zero force
            ],
            dtype=np.float32,
        )
        candidate_forces = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        (
            dense,
            _,
            active,
            total,
            method,
            mapped,
            _,
            _mapped_force_count,
            _mapped_force_explicit_count,
        ) = SofaWallForceInfo._map_contact_records_to_wall_segments(
            contact_records=records,
            wall_vertices=wall_vertices,
            wall_triangles=wall_triangles,
            wall_centroids=wall_centroids,
            candidate_forces=candidate_forces,
            candidate_positions=candidate_positions,
            contact_epsilon=1e-7,
        )
        self.assertEqual(method, "contact_element_triangle_id")
        self.assertEqual(mapped, 1)
        self.assertEqual(active.shape[0], 1)
        self.assertTrue(np.allclose(dense[0], np.asarray([0.0, -1.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(total, np.asarray([0.0, -1.0, 0.0], dtype=np.float32)))

    def test_cached_triangle_mapping_uses_neighbor_indices(self) -> None:
        wall_vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        wall_triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        wall_centroids = wall_vertices[wall_triangles].mean(axis=1)
        # Cache knows index 4, but current force is at index 5.
        cache = {4: 1}
        candidate_positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [10.1, 0.1, 0.0],
                [10.2, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        candidate_forces = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        dense, ids, active, total, method, mapped, explicit_ratio = (
            SofaWallForceInfo._map_force_points_with_cached_triangles(
                wall_vertices=wall_vertices,
                wall_triangles=wall_triangles,
                wall_centroids=wall_centroids,
                candidate_forces=candidate_forces,
                candidate_positions=candidate_positions,
                contact_epsilon=1e-7,
                force_idx_to_wall_triangle=cache,
                max_surface_distance=2.5,
                neighbor_window=2,
                allow_surface_fallback=False,
            )
        )
        self.assertEqual(method, "cached_contact_triangle_id")
        self.assertEqual(mapped, 1)
        self.assertAlmostEqual(explicit_ratio, 1.0, places=6)
        self.assertTrue(np.array_equal(ids, np.asarray([1], dtype=np.int32)))
        self.assertTrue(np.allclose(active[0], np.asarray([0.0, -3.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(dense[1], np.asarray([0.0, -3.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(total, np.asarray([0.0, -3.0, 0.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
