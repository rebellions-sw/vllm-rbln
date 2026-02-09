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

import sys

LICENSE_HEADER_PREFIX = "# Copyright 2025 Rebellions Inc."


def check_license_header(file_path):
    with open(file_path, encoding='UTF-8') as file:
        lines = file.readlines()
        if not lines:
            # Empty file like __init__.py
            return True
        for line in lines:
            if line.strip().startswith(LICENSE_HEADER_PREFIX):
                return True
    return False


def main():
    files_with_missing_header = []
    for file_path in sys.argv[1:]:
        if not check_license_header(file_path):
            files_with_missing_header.append(file_path)

    if files_with_missing_header:
        print("The following files are missing the RBLN License header:")
        for file_path in files_with_missing_header:
            print(f"  {file_path}")

    sys.exit(1 if files_with_missing_header else 0)


if __name__ == "__main__":
    main()