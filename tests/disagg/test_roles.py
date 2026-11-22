import unittest

from minisgl.disagg.role import WorkerRole, require_serving_role


class TestRoles(unittest.TestCase):
    def test_default_unified(self):
        require_serving_role("unified")
        require_serving_role(WorkerRole.UNIFIED)

    def test_reserved_roles_give_phase2_message(self):
        for role in (WorkerRole.PREFILL, WorkerRole.DECODE):
            with self.subTest(role=role), self.assertRaisesRegex(NotImplementedError, "Phase 2"):
                require_serving_role(role)

    def test_unknown_role(self):
        with self.assertRaises(ValueError):
            require_serving_role("invalid")


if __name__ == "__main__":
    unittest.main()
