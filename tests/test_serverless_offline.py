"""
Test the POPROX endpoint running under Serverless Offline.
"""

import logging
import sys
from pathlib import Path
from threading import Condition, Lock, Thread

import requests
from pexpect import EOF, spawn
from pytest import fail, fixture, mark

logger = logging.getLogger(__name__)


@fixture(scope="module")
def sl_listener():
    """
    Fixture that starts and stops serverless offline to test endpoint responses.
    """

    thread = ServerlessBackground()
    thread.start()
    try:
        with thread.lock:
            if thread.ready.wait(10):
                logger.info("ready for tests")
                yield
            else:
                fail("serverless timed out")
    finally:
        thread.proc.sendintr()


class ServerlessBackground(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = Lock()
        self.ready = Condition(self.lock)

    def run(self):
        logger.info("starting serverless")
        self.proc = spawn("npx serverless offline start", logfile=sys.stdout.buffer)
        self.proc.expect(r"Server ready:")
        logger.info("server ready")
        with self.lock:
            self.ready.notify_all()
        self.proc.expect(EOF)


@mark.serverless
def test_basic_request(sl_listener):
    test_dir = Path(__file__)
    req_f = test_dir.parent / "basic-request.json"
    req_body = req_f.read_text()

    logger.info("sending request")
    res = requests.post("http://localhost:3000", req_body)
    assert res.status_code == 200
    logger.info("response: %s", res.text)
    body = res.json()
    assert "recommendations" in body
    assert len(body["recommendations"]) > 0
