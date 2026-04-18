from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from steve_recommender.evaluation.sofa_force_monitor import (
    SofaForceMonitorRuntime,
    resolve_monitor_plugin_path,
)


class ResolveMonitorPluginPathTests(unittest.TestCase):
    def test_resolve_prefers_override_then_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            override = tmp_path / "override.so"
            env = tmp_path / "env.so"
            override.write_text("", encoding="utf-8")
            env.write_text("", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"STEVE_WALL_FORCE_MONITOR_PLUGIN": str(env)},
                clear=False,
            ):
                resolved = resolve_monitor_plugin_path(str(override))
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved, override.resolve())

                resolved_env = resolve_monitor_plugin_path(None)
                self.assertIsNotNone(resolved_env)
                self.assertEqual(resolved_env, env.resolve())


class SofaForceMonitorRuntimeTests(unittest.TestCase):
    class _Data:
        def __init__(self, value):
            self.value = value

    class _Lcp:
        def __init__(self):
            self.build_lcp = SofaForceMonitorRuntimeTests._Data(False)
            self.computeConstraintForces = SofaForceMonitorRuntimeTests._Data(False)

    class _Root:
        def __init__(self, *, with_monitor: bool = False):
            self.LCP = SofaForceMonitorRuntimeTests._Lcp()
            if with_monitor:
                self.wire_wall_force_monitor = object()

        def addObject(self, *_args, **_kwargs):
            return None

    class _Sim:
        def __init__(self, *, with_monitor: bool = False):
            self.root = SofaForceMonitorRuntimeTests._Root(with_monitor=with_monitor)

    def test_intrusive_mode_sets_lcp_flags(self) -> None:
        sim = self._Sim()
        runtime = SofaForceMonitorRuntime(
            mode="intrusive_lcp",
            contact_epsilon=1e-7,
            plugin_path=None,
        )
        status = runtime.ensure(sim)
        self.assertTrue(status.configured)
        self.assertEqual(status.source, "intrusive_lcp")
        self.assertTrue(sim.root.LCP.build_lcp.value)
        self.assertTrue(sim.root.LCP.computeConstraintForces.value)

    def test_passive_mode_reports_missing_plugin(self) -> None:
        sim = self._Sim()
        runtime = SofaForceMonitorRuntime(
            mode="passive",
            contact_epsilon=1e-7,
            plugin_path="/tmp/this-plugin-does-not-exist.so",
        )
        with mock.patch.dict(os.environ, {"STEVE_WALL_FORCE_MONITOR_PLUGIN": ""}, clear=False):
            status = runtime.ensure(sim)
        self.assertFalse(status.configured)
        self.assertIn(
            status.source,
            {
                "passive_plugin_missing",
                "passive_monitor_missing_after_attach",
                "passive_monitor_attach_failed",
            },
        )

    def test_passive_mode_with_existing_monitor_skips_contact_listener_by_default(self) -> None:
        sim = self._Sim(with_monitor=True)
        with mock.patch.dict(os.environ, {"STEVE_FORCE_ENABLE_CONTACT_LISTENER": "0"}, clear=False):
            runtime = SofaForceMonitorRuntime(
                mode="passive",
                contact_epsilon=1e-7,
                plugin_path=None,
            )
            status = runtime.ensure(sim)
        self.assertTrue(status.configured)
        self.assertEqual(status.source, "passive_monitor")
        self.assertEqual(status.error, "")


if __name__ == "__main__":
    unittest.main()
