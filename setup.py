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
import os

from setuptools import find_packages, setup
from setuptools_scm import get_version

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_vllm_version() -> str:
    version = get_version(write_to="vllm_rbln/_version.py")
    return version


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> list[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r"):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_rbln.")
    return requirements


setup(name="vllm-rbln",
      version=get_vllm_version(),
      author="Rebellions Inc.",
      author_email="support@rebellions.ai",
      description="vLLM plugin for RBLN NPU",
      long_description=read_readme(),
      long_description_content_type="text/markdown",
      url="https://github.com/rebellions-sw/vllm-rbln",
      project_urls={
          "Homepage": "https://rebellions.ai",
          "Documentation": "https://docs.rbln.ai",
          "Repository": "https://github.com/rebellions-sw/vllm-rbln"
      },
      classifiers=[
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Programming Language :: Python :: 3.12",
          "Programming Language :: Python :: 3.13",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: Apache Software License",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Information Analysis",
      ],
      packages=find_packages(),
      python_requires=">=3.9",
      install_requires=get_requirements(),
      entry_points={
          "vllm.platform_plugins": ["rbln = vllm_rbln:register"],
          "vllm.general_plugins":
          ["rbln_new_models = vllm_rbln:register_model"]
      })
