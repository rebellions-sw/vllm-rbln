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

import logging
import logging.handlers

from vllm_rbln.logger import init_logger


def test_logger():
    logger = init_logger(__name__)
    assert logger.name.strip().startswith("vllm"), (
        f"Expected logger name to start with 'vllm', got {logger.name}"
    )


def test_logger_has_once_methods():
    """Test that init_logger patches debug_once, info_once, warning_once."""
    logger = init_logger("test_once_methods")
    assert hasattr(logger, "debug_once")
    assert hasattr(logger, "info_once")
    assert hasattr(logger, "warning_once")
    assert callable(logger.debug_once)
    assert callable(logger.info_once)
    assert callable(logger.warning_once)


def test_logger_once_only_logs_once():
    """Test that *_once methods only log the same message once."""
    logger = init_logger("test_log_once")
    logger.setLevel(logging.DEBUG)

    handler = logging.handlers.MemoryHandler(capacity=100)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.info_once("unique message 12345")
        logger.info_once("unique message 12345")
        logger.info_once("unique message 12345")
        # Flush to ensure records are captured
        handler.flush()
        msgs = [
            r.getMessage()
            for r in handler.buffer
            if "unique message 12345" in r.getMessage()
        ]
        assert len(msgs) == 1, f"Expected 1 log, got {len(msgs)}"
    finally:
        logger.removeHandler(handler)


def test_enable_trace_function_call(tmp_path):
    """Test enable_trace_function_call sets sys.settrace and writes trace logs."""
    import sys

    from vllm_rbln.logger import enable_trace_function_call

    log_file = tmp_path / "trace.log"
    old_trace = sys.gettrace()
    try:
        enable_trace_function_call(str(log_file))
        assert sys.gettrace() is not None

        # Trigger a traced function call to verify file output
        from vllm_rbln.logger import init_logger

        init_logger("trace_test_verification")

        assert log_file.exists(), "Trace log file should be created"
        content = log_file.read_text()
        assert len(content) > 0, "Trace log should contain entries"
        assert "Call to" in content or "Return from" in content, (
            "Trace log should contain call/return entries"
        )
    finally:
        sys.settrace(old_trace)
