import os
import unittest
from unittest.mock import patch

from services.ros_environment import (
    DEFAULT_TROS_SETUP,
    TROS_ENV_READY_KEY,
    TROS_ENV_SIGNATURE_KEY,
    can_reuse_tros_env,
    seal_tros_env,
)


def valid_tros_environment():
    env = {TROS_ENV_READY_KEY: DEFAULT_TROS_SETUP}
    for key in ("AMENT_PREFIX_PATH", "PATH", "PYTHONPATH", "LD_LIBRARY_PATH"):
        env[key] = os.pathsep.join(("/opt/tros/humble/lib", "/opt/ros/humble/lib"))
    env["ROS_LOCALHOST_ONLY"] = "1"
    return env


class RosEnvironmentReuseTests(unittest.TestCase):
    @patch("services.ros_environment._default_setup_exists", return_value=True)
    def test_trusted_complete_environment_can_be_reused(self, _exists):
        env = valid_tros_environment()
        seal_tros_env(env)

        self.assertIn(TROS_ENV_SIGNATURE_KEY, env)
        self.assertTrue(can_reuse_tros_env(env, DEFAULT_TROS_SETUP))
        self.assertFalse(can_reuse_tros_env(env, "/root/wall-e-bt/install/setup.bash"))

    @patch("services.ros_environment._default_setup_exists", return_value=True)
    def test_missing_marker_or_required_ros_path_refuses_reuse(self, _exists):
        for mutate in (
            lambda env: env.pop(TROS_ENV_READY_KEY),
            lambda env: env.__setitem__("PYTHONPATH", "/opt/tros/humble/lib"),
        ):
            env = valid_tros_environment()
            env[TROS_ENV_SIGNATURE_KEY] = "stale"
            mutate(env)
            seal_tros_env(env)

            self.assertNotIn(TROS_ENV_SIGNATURE_KEY, env)
            self.assertFalse(can_reuse_tros_env(env, DEFAULT_TROS_SETUP))

    @patch("services.ros_environment._default_setup_exists", return_value=True)
    def test_child_environment_changes_force_fallback(self, _exists):
        for mutate in (
            lambda env: env.__setitem__("PATH", env["PATH"] + os.pathsep + "/changed"),
            lambda env: env.__setitem__("ROS_DOMAIN_ID", "42"),
            lambda env: env.__setitem__(TROS_ENV_SIGNATURE_KEY, "invalid"),
        ):
            env = valid_tros_environment()
            seal_tros_env(env)
            mutate(env)

            self.assertFalse(can_reuse_tros_env(env, DEFAULT_TROS_SETUP))

    @patch("services.ros_environment._default_setup_exists", return_value=False)
    def test_missing_setup_script_refuses_reuse(self, _exists):
        env = valid_tros_environment()
        seal_tros_env(env)

        self.assertNotIn(TROS_ENV_SIGNATURE_KEY, env)
        self.assertFalse(can_reuse_tros_env(env, DEFAULT_TROS_SETUP))


if __name__ == "__main__":
    unittest.main()
