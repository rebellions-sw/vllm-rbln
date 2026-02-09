# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_platform_plugins():
    """This test requires RBLN NPU hardware (runs LLM())."""
    import os
    import runpy

    current_file = __file__

    example_file = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        ),
        "examples",
        "experimental/offline_inference_basic.py",
    )
    runpy.run_path(example_file)

    # check if the plugin is loaded correctly
    from vllm.platforms import _init_trace, current_platform

    assert current_platform.plugin_name == "rbln", (
        f"Expected DummyDevice, got {current_platform.plugin_name}, "
        "possibly because current_platform is imported before the plugin"
        f" is loaded. The first import:\n{_init_trace}"
    )
