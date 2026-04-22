from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from steve_recommender.eval_v2.discovery import (
    FileBasedExplicitPolicyDiscovery,
    FileBasedWireRegistryDiscovery,
)
from steve_recommender.eval_v2.models import AgentRef, PolicySpec, WireRef


class FileBasedWireRegistryDiscoveryTests(unittest.TestCase):
    def _write_registry(self, root: Path) -> Path:
        registry_root = root / "wire_registry"
        registry_root.mkdir(parents=True)

        agent_a_dir = (
            registry_root
            / "steve_default"
            / "wire_versions"
            / "standard_j"
            / "agents"
            / "archvar_original_best"
        )
        agent_b_dir = (
            registry_root
            / "universal_ii"
            / "wire_versions"
            / "standard_j"
            / "agents"
            / "archvar_universalII_best"
        )
        agent_a_dir.mkdir(parents=True)
        agent_b_dir.mkdir(parents=True)
        (agent_a_dir / "checkpoints").mkdir(parents=True)
        (agent_b_dir / "checkpoints").mkdir(parents=True)

        checkpoint_a = agent_a_dir / "checkpoints" / "best_checkpoint.everl"
        checkpoint_a.write_text("checkpoint-a", encoding="utf-8")
        checkpoint_b = agent_b_dir / "checkpoints" / "best_checkpoint.everl"

        agent_a_json = agent_a_dir / "agent.json"
        agent_a_json.write_text(
            json.dumps(
                {
                    "name": "archvar_original_best",
                    "checkpoint": "checkpoints/best_checkpoint.everl",
                    "run_dir": "checkpoints",
                    "tool": "steve_default/standard_j",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        agent_b_json = agent_b_dir / "agent.json"
        agent_b_json.write_text(
            json.dumps(
                {
                    "name": "archvar_universalII_best",
                    "checkpoint": "checkpoints/best_checkpoint.everl",
                    "run_dir": "checkpoints",
                    "tool": "universal_ii/standard_j",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        index_payload = {
            "schema_version": 1,
            "wires": {
                "steve_default/standard_j": {
                    "model": "steve_default",
                    "name": "standard_j",
                    "tool_ref": "steve_default/standard_j",
                    "wire_dir": "steve_default/wire_versions/standard_j",
                    "agents_dir": "steve_default/wire_versions/standard_j/agents",
                },
                "steve_default/straight": {
                    "model": "steve_default",
                    "name": "straight",
                    "tool_ref": "steve_default/straight",
                    "wire_dir": "steve_default/wire_versions/straight",
                    "agents_dir": None,
                },
                "universal_ii/standard_j": {
                    "model": "universal_ii",
                    "name": "standard_j",
                    "tool_ref": "universal_ii/standard_j",
                    "wire_dir": "universal_ii/wire_versions/standard_j",
                    "agents_dir": "universal_ii/wire_versions/standard_j/agents",
                },
            },
            "agents": {
                "steve_default/standard_j:archvar_original_best": {
                    "agent_ref": "steve_default/standard_j:archvar_original_best",
                    "model": "steve_default",
                    "wire": "standard_j",
                    "name": "archvar_original_best",
                    "agent_json_path": "steve_default/wire_versions/standard_j/agents/archvar_original_best/agent.json",
                    "checkpoint_path": "steve_default/wire_versions/standard_j/agents/archvar_original_best/checkpoints/best_checkpoint.everl",
                    "run_dir": "steve_default/wire_versions/standard_j/agents/archvar_original_best/checkpoints",
                    "checkpoint_exists": True,
                },
                "universal_ii/standard_j:archvar_universalII_best": {
                    "agent_ref": "universal_ii/standard_j:archvar_universalII_best",
                    "model": "universal_ii",
                    "wire": "standard_j",
                    "name": "archvar_universalII_best",
                    "agent_json_path": "universal_ii/wire_versions/standard_j/agents/archvar_universalII_best/agent.json",
                    "checkpoint_path": str(checkpoint_b),
                    "run_dir": "universal_ii/wire_versions/standard_j/agents/archvar_universalII_best/checkpoints",
                    "checkpoint_exists": False,
                },
            },
        }
        registry_path = registry_root / "index.json"
        registry_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
        return registry_path

    def test_list_execution_wires_returns_all_registry_wires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_registry(Path(tmp))
            discovery = FileBasedWireRegistryDiscovery(registry_path=registry_path)

            wires = discovery.list_execution_wires()

        self.assertEqual(
            wires,
            (
                WireRef(model="steve_default", wire="standard_j"),
                WireRef(model="steve_default", wire="straight"),
                WireRef(model="universal_ii", wire="standard_j"),
            ),
        )

    def test_list_startable_wires_returns_only_wires_with_loadable_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_registry(Path(tmp))
            discovery = FileBasedWireRegistryDiscovery(registry_path=registry_path)

            wires = discovery.list_startable_wires()

        self.assertEqual(
            wires,
            (WireRef(model="steve_default", wire="standard_j"),),
        )

    def test_list_registry_policies_maps_loadable_agents_to_policy_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_registry(Path(tmp))
            discovery = FileBasedWireRegistryDiscovery(registry_path=registry_path)

            policies = discovery.list_registry_policies()

            self.assertEqual(len(policies), 1)
            self.assertTrue(all(isinstance(policy, PolicySpec) for policy in policies))

            policy = policies[0]
            self.assertEqual(policy.name, "archvar_original_best")
            self.assertEqual(policy.source, "registry")
            self.assertEqual(
                policy.trained_on_wire,
                WireRef(model="steve_default", wire="standard_j"),
            )
            self.assertEqual(
                policy.registry_agent,
                AgentRef(
                    wire=WireRef(model="steve_default", wire="standard_j"),
                    agent="archvar_original_best",
                ),
            )
            self.assertEqual(
                policy.agent_ref,
                "steve_default/standard_j:archvar_original_best",
            )
            self.assertTrue(policy.checkpoint_path.exists())
            self.assertTrue(policy.metadata_path.exists())
            self.assertEqual(policy.run_dir, policy.metadata_path.parent / "checkpoints")

    def test_list_registry_policies_can_filter_by_execution_wire(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_registry(Path(tmp))
            discovery = FileBasedWireRegistryDiscovery(registry_path=registry_path)

            same_wire = discovery.list_registry_policies(
                execution_wire=WireRef(model="steve_default", wire="standard_j")
            )
            other_wire = discovery.list_registry_policies(
                execution_wire=WireRef(model="steve_default", wire="straight")
            )

        self.assertEqual(len(same_wire), 1)
        self.assertEqual(other_wire, ())

    def test_resolve_policy_from_agent_ref_returns_policy_and_raises_for_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_registry(Path(tmp))
            discovery = FileBasedWireRegistryDiscovery(registry_path=registry_path)
            agent_ref = AgentRef(
                wire=WireRef(model="steve_default", wire="standard_j"),
                agent="archvar_original_best",
            )

            policy = discovery.resolve_policy_from_agent_ref(agent_ref)

            with self.assertRaises(KeyError):
                discovery.resolve_policy_from_agent_ref(
                    AgentRef(
                        wire=WireRef(model="steve_default", wire="standard_j"),
                        agent="missing_agent",
                    )
                )

        self.assertEqual(policy.name, "archvar_original_best")
        self.assertEqual(policy.registry_agent, agent_ref)

    def test_list_explicit_policies_defaults_to_empty_until_explicit_registry_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_registry(Path(tmp))
            discovery = FileBasedWireRegistryDiscovery(registry_path=registry_path)

            policies = discovery.list_explicit_policies()

        self.assertEqual(policies, ())


class FileBasedExplicitPolicyDiscoveryTests(unittest.TestCase):
    def _write_manifest(self, root: Path) -> Path:
        manifest_root = root / "wire_registry"
        source_agent_dir = manifest_root / "source" / "agents" / "best_agent"
        source_agent_dir.mkdir(parents=True)
        checkpoint_path = source_agent_dir / "best_checkpoint.everl"
        checkpoint_path.write_text("checkpoint-explicit", encoding="utf-8")
        agent_json_path = source_agent_dir / "agent.json"
        agent_json_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "name": "best_agent",
                    "run_dir": str(source_agent_dir),
                    "tool": "steve_default/standard_j",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        manifest_payload = {
            "schema_version": 1,
            "versions": [
                {
                    "new_model": "steve_default",
                    "new_version": "standard_j",
                    "agents": [
                        {
                            "agent_json": str(agent_json_path),
                            "agent_name": "best_agent",
                            "source_checkpoint": str(checkpoint_path),
                            "run_dir": str(source_agent_dir),
                        }
                    ],
                }
            ],
        }
        manifest_path = manifest_root / "archvar_inventory_manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        return manifest_path

    def test_list_explicit_policies_reads_manifest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_manifest(Path(tmp))
            discovery = FileBasedExplicitPolicyDiscovery(manifest_path=manifest_path)

            policies = discovery.list_explicit_policies()
            self.assertEqual(len(policies), 1)
            policy = policies[0]
            self.assertEqual(policy.name, "best_agent")
            self.assertEqual(policy.source, "explicit")
            self.assertEqual(policy.trained_on_wire, WireRef(model="steve_default", wire="standard_j"))
            self.assertEqual(policy.agent_ref, "steve_default/standard_j:best_agent")
            self.assertTrue(policy.checkpoint_path.exists())
            self.assertTrue(policy.metadata_path.exists())

    def test_resolve_policy_from_agent_ref_returns_explicit_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_manifest(Path(tmp))
            discovery = FileBasedExplicitPolicyDiscovery(manifest_path=manifest_path)
            agent_ref = AgentRef(wire=WireRef(model="steve_default", wire="standard_j"), agent="best_agent")

            policy = discovery.resolve_policy_from_agent_ref(agent_ref)

        self.assertEqual(policy.registry_agent, agent_ref)


if __name__ == "__main__":
    unittest.main()
