"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import glob
import stat
import sys
import sysconfig

import codecs
import os
import re
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open("{}/README.md".format(here), "r") as fh:
    long_description = fh.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

INSTALL_REQUIRES = ["addict>=2.4.0",
                    "texttable>=1.6.3",
                    "scipy>=1.3.2",
                    "networkx>=2.5",
                    "jsonschema==3.2.0",
                    "jstyleson>=0.0.2", # only samples
                    "numpy>=1.19"]

python_version = sys.version_info[:2]
if python_version < (3, 6):
    print("Only Python >= 3.6 is supported")
    sys.exit(0)

if not os.environ.get('NNCF_SKIP_INSTALLING_TF'):
    INSTALL_REQUIRES.extend(["tensorflow==2.4.0"])
else:
    print("Skipping tensorflow installation for NNCF.")
    DEPENDENCY_LINKS = []


EXTRAS_REQUIRE = {
    "tests": [
        "pytest"],
    "docs": []
}

setup(
    name="nncf_tensorflow",
    version=find_version(os.path.join(here, "nncf/version.py")),
    author="Intel",
    author_email="alexander.kozlov@intel.com",
    description="Neural Networks Compression Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openvinotoolkit/nncf",
    packages=find_packages(exclude=["tests", "tests.*",
                                    "examples", "examples.*",
                                    "tools", "tools.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    keywords=["compression", "quantization", "sparsity", "mixed-precision-training",
              "quantization-aware-training", "hawq", "classification",
              "pruning", "object-detection", "semantic-segmentation"],
    include_package_data=True
)
