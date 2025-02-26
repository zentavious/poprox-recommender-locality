"""
Support code for testing the poprox-recommender components and service.

This lives in the main package so it can be easily imported from any of the
tests, regardless of their subdirectories.
"""

import base64
import gzip
import json
import logging
import os
import subprocess as sp
from collections.abc import Generator
from signal import SIGINT
from time import sleep
from typing import Protocol

import requests
from pytest import fixture

from poprox_concepts.api.recommendations import RecommendationRequest, RecommendationResponse

logger = logging.getLogger(__name__)


class TestService(Protocol):
    """
    Interface for test services bridges that accept requests and return
    responses. It abstracts direct local runs, Docker, and AWS Lambda test
    endpoints.
    """

    def request(self, req: RecommendationRequest | str, pipeline: str) -> RecommendationResponse:
        """
        Request recommendations from the recommender.
        """
        raise NotImplementedError()


class InProcessTestService:
    """
    Test service that directly runs the request handler in-process.
    """

    def request(
        self, req: RecommendationRequest | str, pipeline: str, compress: bool = False
    ) -> RecommendationResponse:
        # defer to here so we don't always import the handler
        from poprox_recommender.handler import generate_recs

        if not isinstance(req, str):
            req = req.model_dump_json()

        req_txt = json.dumps(json.loads(req))

        if compress:
            event = {
                "headers": {"Content-Encoding": "gzip", "Content-Type": "application/json"},
                "queryStringParameters": {"pipeline": pipeline},
                "body": base64.encodebytes(gzip.compress(req_txt.encode())).decode("ascii"),
                "isBase64Encoded": True,
            }
        else:
            event = {
                "headers": {},
                "queryStringParameters": {"pipeline": pipeline},
                "body": req_txt,
                "isBase64Encoded": False,
            }

        res = generate_recs(event, {})
        return RecommendationResponse.model_validate_json(res["body"])


class DockerTestService:
    """
    Test service that directly runs the request handler in-process.
    """

    url: str

    def __init__(self, url: str):
        self.url = url

    def request(
        self, req: RecommendationRequest | str, pipeline: str, compress: bool = False
    ) -> RecommendationResponse:
        if not isinstance(req, str):
            req = req.model_dump_json()

        req_txt = json.dumps(json.loads(req))

        if compress:
            event = {
                "headers": {"Content-Encoding": "gzip", "Content-Type": "application/json"},
                "queryStringParameters": {"pipeline": pipeline},
                "body": base64.encodebytes(gzip.compress(req_txt.encode())).decode("ascii"),
                "isBase64Encoded": True,
            }
        else:
            event = {
                "headers": {},
                "queryStringParameters": {"pipeline": pipeline},
                "body": req_txt,
                "isBase64Encoded": False,
            }

        result = requests.post(self.url, json=event)
        res_data = result.json()
        if result.status_code != 200:
            logger.error("Lambda function failed with type %s", res_data["errorType"])
            logger.info("Lambda error message: %s", res_data["errorMessage"])
            if res_data["stackTrace"]:
                logger.info("Stack trace:%s", [f"\n  {i}: {t}" for (i, t) in enumerate(res_data["stackTrace"], 1)])
            raise AssertionError("lambda function failed")

        if res_data["statusCode"] != 200:
            logger.error("HTTP succeeded but lambda failed with code %s", res_data["statusCode"])
            logger.info("result: %s", json.dumps(res_data, indent=2))
            raise AssertionError("lambda request failed")

        return RecommendationResponse.model_validate_json(res_data["body"])


def local_service_impl() -> Generator[TestService, None, None]:
    yield InProcessTestService()


local_service = fixture(local_service_impl)


def docker_service_impl() -> Generator[TestService, None, None]:
    logger.info("building docker container")
    sp.check_call(["docker-buildx", "build", "-t", "poprox-recommender:test", "."])
    logger.info("starting docker container")
    proc = sp.Popen(["docker", "run", "--rm", "-p", "18080:8080", "poprox-recommender:test"])
    sleep(1)
    try:
        yield DockerTestService("http://localhost:18080/2015-03-31/functions/function/invocations")
    finally:
        logger.info("stopping docker container")
        proc.send_signal(SIGINT)
        proc.wait()


docker_service = fixture(scope="session")(docker_service_impl)


@fixture(scope="session")
def auto_service() -> Generator[TestService, None, None]:
    backend = os.environ.get("POPROX_TEST_TARGET", "local")
    if backend == "local":
        yield from local_service_impl()
    elif backend == "docker-auto":
        yield from docker_service_impl()
    elif backend == "docker":
        # use already-running docker
        port = os.environ.get("POPROX_TEST_PORT", "9000")
        yield DockerTestService(f"http://localhost:{port}/2015-03-31/functions/function/invocations")
