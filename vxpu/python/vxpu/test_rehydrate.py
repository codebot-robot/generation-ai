# Copyright 2026 Google LLC
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

"""Unit tests for executor direct URL resolution, caching, and retry in rehydrate.py."""

import http.server
import re
import socketserver
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

# Mock torch if not installed in this environment
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.export.passes"] = MagicMock()

from vxpu import rehydrate


def _make_gcs_url(base_url, offset_sec=0, expires_sec=604800):
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_sec)
    date_str = dt.strftime("%Y%m%dT%H%M%SZ")
    return f"{base_url}/gcs/blob?X-Goog-Date={date_str}&X-Goog-Expires={expires_sec}&signature=secret123"


class MockBlobServer(http.server.BaseHTTPRequestHandler):
    """Mock server serving modelstore endpoints and GCS direct endpoints."""

    def log_message(self, format, *args):
        pass  # quiet test logs

    def do_HEAD(self):
        self.server.head_requests.append(self.path)
        if self.server.fail_head_once:
            self.server.fail_head_once = False
            self.send_response(self.server.head_error_status)
            if hasattr(self.server, "head_retry_after") and self.server.head_retry_after:
                self.send_header("Retry-After", str(self.server.head_retry_after))
            self.end_headers()
            return
        if self.path.startswith("/blobs/with-direct"):
            self.send_response(200)
            self.send_header("X-Modelstore-Direct-Url", self.server.direct_url)
            self.send_header("Content-Length", str(len(self.server.blob_data)))
            self.end_headers()
        elif self.path.startswith("/blobs/no-direct"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.server.blob_data)))
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        self.server.get_requests.append(self.path)
        # Check redirect parameter
        if "?redirect=1" in self.path or "&redirect=1" in self.path:
            if self.path.startswith("/blobs/with-redirect"):
                self.send_response(307)
                self.send_header("Location", self.server.direct_url)
                self.send_header("X-Modelstore-Direct-Url", self.server.direct_url)
                self.end_headers()
                return

        # Direct GCS range read endpoint
        if self.path.startswith("/gcs/"):
            if self.path.startswith("/gcs/notfound"):
                self.send_response(404)
                self.end_headers()
                return

            # GCS answers expired V4 signed URLs with 400 ExpiredToken
            if self.server.return_400_once and not self.server.returned_400:
                self.server.returned_400 = True
                self.send_response(400)
                self.send_header("Content-Type", "application/xml")
                self.end_headers()
                self.wfile.write(
                    b"<?xml version='1.0' encoding='UTF-8'?>"
                    b"<Error><Code>ExpiredToken</Code>"
                    b"<Message>The provided token has expired.</Message></Error>"
                )
                return

            if self.server.return_403_once and not self.server.returned_403:
                self.server.returned_403 = True
                self.send_response(403)
                self.end_headers()
                return

            self._serve_range(self.server.blob_data)
            return

        # Modelstore direct range read endpoint
        if self.path.startswith("/blobs/"):
            self._serve_range(self.server.blob_data)
            return

        self.send_response(404)
        self.end_headers()

    def _serve_range(self, data):
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        match = re.match(r"bytes=(\d+)-(\d+)", rng)
        if not match:
            self.send_response(416)
            self.end_headers()
            return
        start = int(match.group(1))
        end = int(match.group(2))
        chunk = data[start:end + 1]

        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


class TestRehydrateDirectURL(unittest.TestCase):
    def setUp(self):
        rehydrate.clear_url_cache()
        self.server = socketserver.TCPServer(("127.0.0.1", 0), MockBlobServer)
        self.server.blob_data = b"0123456789abcdefghijklmnopqrstuvwxyz"
        self.server.head_requests = []
        self.server.get_requests = []
        self.server.return_400_once = False
        self.server.returned_400 = False
        self.server.return_403_once = False
        self.server.returned_403 = False
        self.server.fail_head_once = False
        self.server.head_error_status = 500
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.server.direct_url = _make_gcs_url(self.base_url)

        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_direct_url_via_head_cached_per_file(self):
        source_url = f"{self.base_url}/blobs/with-direct"

        buf1 = bytearray(10)
        view1 = memoryview(buf1)
        rehydrate._read_range(source_url, 0, 10, view1)
        self.assertEqual(bytes(buf1), b"0123456789")

        # Head was called once to ask for direct URL
        self.assertEqual(len(self.server.head_requests), 1)
        # GCS direct URL was used for GET range read, not /blobs/
        gcs_gets = [p for p in self.server.get_requests if p.startswith("/gcs/")]
        self.assertEqual(len(gcs_gets), 1)

        # Second range read on the same file should use cached direct URL without new HEAD
        buf2 = bytearray(6)
        view2 = memoryview(buf2)
        rehydrate._read_range(source_url, 10, 6, view2)
        self.assertEqual(bytes(buf2), b"abcdef")

        # Head requests count should still be 1!
        self.assertEqual(len(self.server.head_requests), 1)
        # Second GET went straight to GCS
        gcs_gets = [p for p in self.server.get_requests if p.startswith("/gcs/")]
        self.assertEqual(len(gcs_gets), 2)

    def test_direct_url_via_redirect(self):
        source_url = f"{self.base_url}/blobs/with-redirect"

        buf = bytearray(4)
        view = memoryview(buf)
        rehydrate._read_range(source_url, 4, 4, view)
        self.assertEqual(bytes(buf), b"4567")

        gcs_gets = [p for p in self.server.get_requests if p.startswith("/gcs/")]
        self.assertEqual(len(gcs_gets), 1)

    def test_fallback_when_no_direct_url_cached_with_ttl(self):
        source_url = f"{self.base_url}/blobs/no-direct"

        buf = bytearray(5)
        view = memoryview(buf)
        rehydrate._read_range(source_url, 0, 5, view)
        self.assertEqual(bytes(buf), b"01234")

        # Head request occurred once
        self.assertEqual(len(self.server.head_requests), 1)
        # GET went through /blobs/ directly
        blob_gets = [p for p in self.server.get_requests if p.startswith("/blobs/no-direct")]
        self.assertEqual(len(blob_gets), 1)

        # Second read uses cached fallback within TTL without new HEAD
        buf2 = bytearray(3)
        rehydrate._read_range(source_url, 5, 3, memoryview(buf2))
        self.assertEqual(bytes(buf2), b"567")
        self.assertEqual(len(self.server.head_requests), 1)

    def test_transient_failure_on_head_is_not_cached(self):
        source_url = f"{self.base_url}/blobs/with-direct"
        self.server.fail_head_once = True
        self.server.head_error_status = 500

        # First read fails HEAD with 500, falls back to source_url without caching
        buf1 = bytearray(4)
        rehydrate._read_range(source_url, 0, 4, memoryview(buf1))
        self.assertEqual(bytes(buf1), b"0123")
        self.assertEqual(len(self.server.head_requests), 1)

        # Second read tries HEAD again (not pinned to proxy mode forever) and gets direct URL!
        buf2 = bytearray(4)
        rehydrate._read_range(source_url, 4, 4, memoryview(buf2))
        self.assertEqual(bytes(buf2), b"4567")
        self.assertEqual(len(self.server.head_requests), 2)
        gcs_gets = [p for p in self.server.get_requests if p.startswith("/gcs/")]
        self.assertEqual(len(gcs_gets), 1)

    def test_retry_on_400_expired_token_refreshes_direct_url(self):
        source_url = f"{self.base_url}/blobs/with-direct"
        self.server.return_400_once = True
        self.server.returned_400 = False

        buf = bytearray(8)
        view = memoryview(buf)
        rehydrate._read_range(source_url, 2, 8, view)
        self.assertEqual(bytes(buf), b"23456789")

        # When GCS answered 400 ExpiredToken on the first attempt, executor re-asked modelstore (HEAD count = 2)
        # and retried reading GCS successfully!
        self.assertEqual(len(self.server.head_requests), 2)
        self.assertTrue(self.server.returned_400)

    def test_retry_on_403_refreshes_direct_url(self):
        source_url = f"{self.base_url}/blobs/with-direct"
        self.server.return_403_once = True
        self.server.returned_403 = False

        buf = bytearray(8)
        view = memoryview(buf)
        rehydrate._read_range(source_url, 2, 8, view)
        self.assertEqual(bytes(buf), b"23456789")

        # When GCS answered 403 on the first attempt, executor re-asked modelstore (HEAD count = 2)
        # and retried reading GCS successfully!
        self.assertEqual(len(self.server.head_requests), 2)
        self.assertTrue(self.server.returned_403)

    def test_expired_url_refreshes_automatically(self):
        source_url = f"{self.base_url}/blobs/with-direct"
        # Set an expired direct URL initially (well in the past)
        self.server.direct_url = _make_gcs_url(self.base_url, offset_sec=-1000000, expires_sec=60)

        # First read gets the expired URL and caches it (but expired)
        resolved_first = rehydrate._resolve_url(source_url)
        self.assertEqual(len(self.server.head_requests), 1)

        # Update server direct URL to new valid URL
        new_url = _make_gcs_url(self.base_url, offset_sec=0, expires_sec=604800)
        self.server.direct_url = new_url

        # Next resolve checks expiry, detects it is expired, and asks again!
        resolved_second = rehydrate._resolve_url(source_url)
        self.assertEqual(len(self.server.head_requests), 2)
        self.assertEqual(resolved_second, new_url)

    def test_signed_url_query_redacted_on_error(self):
        source_url = f"{self.base_url}/blobs/with-direct"
        self.server.direct_url = _make_gcs_url(f"{self.base_url}/gcs/notfound") + "&secret_token=supersecret"

        buf = bytearray(4)
        import requests
        with self.assertRaises(requests.HTTPError) as ctx:
            rehydrate._read_range(source_url, 0, 4, memoryview(buf))

        err_str = str(ctx.exception)
        # Redacted URL must not contain the query string with secret token
        self.assertNotIn("secret_token=supersecret", err_str)
        self.assertNotIn("X-Goog-Expires", err_str)
        self.assertIn("/gcs/notfound", err_str)
        if ctx.exception.response is not None:
            self.assertNotIn("secret_token=supersecret", ctx.exception.response.url)
        if ctx.exception.request is not None:
            self.assertNotIn("secret_token=supersecret", ctx.exception.request.url)

    def test_signed_url_connection_error_redacted(self):
        source_url = f"{self.base_url}/blobs/with-direct"
        # Direct URL points to an unreachable port with sensitive parameters
        self.server.direct_url = "http://127.0.0.1:1/gcs/blob?secret_sig=supersecret&X-Goog-Expires=604800"

        buf = bytearray(4)
        import requests
        with self.assertRaises(requests.RequestException) as ctx:
            rehydrate._read_range(source_url, 0, 4, memoryview(buf))

        err_str = str(ctx.exception)
        self.assertNotIn("secret_sig=supersecret", err_str)
        self.assertNotIn("X-Goog-Expires", err_str)
        self.assertIn("http://127.0.0.1:1/gcs/blob", err_str)

    def test_503_retry_after_caches_fallback_for_duration(self):
        source_url = f"{self.base_url}/blobs/with-direct"
        self.server.fail_head_once = True
        self.server.head_error_status = 503
        self.server.head_retry_after = "2"

        # First read hits 503 on HEAD, falls back to source_url and caches fallback with 2s TTL
        buf1 = bytearray(4)
        rehydrate._read_range(source_url, 0, 4, memoryview(buf1))
        self.assertEqual(bytes(buf1), b"0123")
        self.assertEqual(len(self.server.head_requests), 1)

        # Immediate second read uses cached fallback within 2s without issuing a new HEAD
        buf2 = bytearray(4)
        rehydrate._read_range(source_url, 4, 4, memoryview(buf2))
        self.assertEqual(bytes(buf2), b"4567")
        self.assertEqual(len(self.server.head_requests), 1)

    def test_unparseable_expiry_is_treated_as_expired(self):
        source_url = f"{self.base_url}/blobs/with-direct"
        # Direct URL without X-Goog-Date or X-Goog-Expires
        self.server.direct_url = f"{self.base_url}/gcs/blob?no_expiry=1"

        resolved_first = rehydrate._resolve_url(source_url)
        self.assertEqual(len(self.server.head_requests), 1)

        # An unparseable expiry is treated as expired immediately (not cached forever)
        entry = rehydrate._direct_url_cache.get(source_url)
        self.assertIsNotNone(entry)
        self.assertTrue(entry.is_expired())

        # Next resolve should ask again rather than keeping it forever
        rehydrate._resolve_url(source_url)
        self.assertEqual(len(self.server.head_requests), 2)

    def test_concurrent_threads_share_cached_direct_url(self):
        source_url = f"{self.base_url}/blobs/with-direct"

        def read_part(offset, length):
            buf = bytearray(length)
            view = memoryview(buf)
            rehydrate._read_range(source_url, offset, length, view)
            return bytes(buf)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read_part, i * 4, 4) for i in range(5)]
            results = [f.result() for f in futures]

        self.assertEqual(b"".join(results), b"0123456789abcdefghij")
        # HEAD should be called once across concurrent threads
        self.assertEqual(len(self.server.head_requests), 1)

    def test_concurrent_readers_on_expired_token_share_one_refresh(self):
        source_url = f"{self.base_url}/blobs/with-direct"

        # First read warms cache
        buf = bytearray(4)
        rehydrate._read_range(source_url, 0, 4, memoryview(buf))
        self.assertEqual(len(self.server.head_requests), 1)

        # Configure server to return 400 ExpiredToken once
        self.server.return_400_once = True
        self.server.returned_400 = False

        # Run multiple concurrent readers: all hit 400 on the same cached URL
        def read_part(offset, length):
            b = bytearray(length)
            rehydrate._read_range(source_url, offset, length, memoryview(b))
            return bytes(b)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read_part, i * 4, 4) for i in range(5)]
            results = [f.result() for f in futures]

        self.assertEqual(b"".join(results), b"0123456789abcdefghij")
        # Only ONE new HEAD request should be issued for the refresh, not 5!
        self.assertEqual(len(self.server.head_requests), 2)


if __name__ == "__main__":
    unittest.main()
