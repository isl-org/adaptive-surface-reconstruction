#
# Copyright 2022 Intel (Autonomous Agents Lab)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from setuptools import setup
from distutils.extension import Extension
from glob import glob

name = "adaptivesurfacereconstruction"
version = os.environ.get("VERSION", "unknown")

# Force platform specific wheel
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    # https://stackoverflow.com/a/45150383/1255535
    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    print(
        'Warning: cannot import "wheel" package to build platform-specific wheel'
    )
    print('Install the "wheel" package to fix this warning')
    bdist_wheel = None

cmdclass = {'bdist_wheel': bdist_wheel} if bdist_wheel is not None else dict()

status = setup(
    name=name,
    version=version,
    description="Python bindings for adaptive surface reconstruction",
    packages=['adaptivesurfacereconstruction'],
    cmdclass=cmdclass,
    include_package_data=True,
)
