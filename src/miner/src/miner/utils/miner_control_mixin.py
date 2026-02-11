import asyncio
import base64
import multiprocessing
import time
from typing import Any, Dict, Optional

import aiohttp
from loguru import logger
from common.models.api_models import EnclaveGetKeyIdResponse, EnclaveSignResponse
from miner import settings as miner_settings
from miner.utils.miner_control_protocol_api import start_protocol_server


class MinerControlMixin:
    miner_control_port: int

    _enclave_key_cache: Optional[Dict[str, Any]] = None
    _is_mounted: bool = (
        miner_settings.IS_MOUNTED
    )  # The control server should be used as a gate only when mounted in the electron app
    _electron_version: str | None = miner_settings.ELECTRON_VERSION

    async def _start_control_protocol_server_process(self, port: int | None = None):
        """Start the Miner Control Protocol server in a separate process"""
        if self._is_mounted:
            try:
                target_port = port or self.miner_control_port
                self.miner_control_process = multiprocessing.Process(
                    target=start_protocol_server, args=(target_port,), daemon=True, name="MinerControlServer"
                )
                self.miner_control_process.start()
                logger.info(f"✅ Protocol server started in separate process (PID: {self.miner_control_process.pid})")
            except Exception as e:
                logger.exception(f"Error starting protocol server process: {e}")
        else:
            print("Miner Control Protocol not enabled")

    async def _stop_control_protocol_server_process(self):
        """Stop the Miner Control Protocol server."""
        if self._is_mounted:
            if self.miner_control_process and self.miner_control_process.is_alive():
                logger.info("Stoppping miner control protocol server process")
                self.miner_control_process.terminate()
                self.miner_control_process.join(timeout=5)

                if self.miner_control_process.is_alive():
                    logger.warning("Miner Control Protocol server did not terminate gracefully, force killing")
                    self.miner_control_process.kill()
                    self.miner_control_process.join()

                logger.info("✅ Miner Control Protocol server stopped")
            self.miner_control_process = None
        else:
            print("Miner Control Protocol not enabled")

    def _control_base_url(self) -> str:
        return f"http://127.0.0.1:{self.miner_control_port}"

    async def _control_get_json(self, path: str, timeout_s: float = 2.0) -> Dict[str, Any]:
        url = self._control_base_url() + path
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"GET {path} failed {r.status}: {text}")
                return await r.json()

    async def _control_post_json(self, path: str, payload: Dict[str, Any], timeout_s: float = 20.0) -> Dict[str, Any]:
        url = self._control_base_url() + path
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"POST {path} failed {r.status}: {text}")
                return await r.json()

    async def wait_for_control_server_up(self, timeout_s: float = 20.0) -> None:
        if not self._is_mounted:
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                await self._control_get_json("/health", timeout_s=2.0)
                return
            except Exception:
                await asyncio.sleep(0.5)
        raise RuntimeError("Miner control server not reachable (/health)")

    async def wait_for_host_connected(self, timeout_s: float = 120.0, require_hello: bool = False) -> Dict[str, Any]:
        if not self._is_mounted:
            return {"ok": True, "connected": False, "skipped": True}

        deadline = time.monotonic() + timeout_s
        last: Optional[Dict[str, Any]] = None

        while time.monotonic() < deadline:
            try:
                last = await self._control_get_json("/health", timeout_s=2.0)
                if last.get("ok") and last.get("connected") is True:
                    if require_hello and last.get("client") is None:
                        await asyncio.sleep(0.5)
                        continue
                    return last
            except Exception:
                pass

            await asyncio.sleep(1.0)

        raise RuntimeError(f"Electron host not connected (last={last})")

    async def ensure_control_ready(self) -> None:
        """Call once at startup before registration/training."""
        if not self._is_mounted:
            return
        await self.wait_for_control_server_up(timeout_s=20.0)
        health = await self.wait_for_host_connected(timeout_s=120.0, require_hello=True)
        logger.info(f"✅ Control host connected: sessionId={health.get('sessionId')}")

    async def enclave_get_key_id(
        self, *, purpose: str, preferred_algorithms=None, dp_keychain=None, cache: bool = True
    ) -> EnclaveGetKeyIdResponse:
        if cache and self._enclave_key_cache is not None:
            return EnclaveGetKeyIdResponse(**self._enclave_key_cache)

        await self.wait_for_host_connected(timeout_s=120.0, require_hello=True)

        payload = {
            "purpose": purpose,
            "preferredAlgorithms": preferred_algorithms or ["ES256"],
            "dpKeychain": dp_keychain,
        }

        for attempt in range(1, 4):
            try:
                res = await self._control_post_json("/enclave/get_key_id", payload, timeout_s=30.0)
                if cache:
                    self._enclave_key_cache = res
                return EnclaveGetKeyIdResponse(**res)
            except Exception as e:
                logger.warning(f"get_key_id failed (attempt {attempt}/3): {e}")
                await asyncio.sleep(2 * attempt)
                await self.wait_for_host_connected(timeout_s=120.0, require_hello=True)

        raise RuntimeError("Failed to get_key_id after retries")

    async def enclave_sign(
        self,
        *,
        key_id: str,
        payload: bytes | str,
        dp_keychain: Optional[bool] = None,
        require_hello: bool = True,
        timeout_s: float = 30.0,
        retries: int = 3,
        challenge_id: Optional[str] = None,
    ) -> EnclaveSignResponse:
        """
        Ask the Electron host enclave to sign `payload` using `key_id` (ES256).

        `payload` may be raw bytes or a base64-encoded string.

        Returns JSON like:
          { "keyId": "...", "signatureDerBase64": "...", "alg": "ES256", "publicKeyX963Base64": optional }
        """
        if not self._is_mounted:
            raise RuntimeError("Miner control disabled; enclave_sign unavailable")

        await self.wait_for_host_connected(timeout_s=120.0, require_hello=require_hello)

        if isinstance(payload, str):
            payload_b64 = payload
        else:
            payload_b64 = base64.b64encode(payload).decode("ascii")

        req: Dict[str, Any] = {
            "keyId": key_id,
            "payloadBase64": payload_b64,
            "alg": "ES256",
        }
        if dp_keychain is not None:
            req["dpKeychain"] = dp_keychain

        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                res = await self._control_post_json("/enclave/sign", req, timeout_s=timeout_s)
                res["challenge_id"] = challenge_id
                return EnclaveSignResponse(**res)
            except Exception as e:
                last_err = e
                logger.warning(f"enclave_sign failed (attempt {attempt}/{retries}): {e}")
                await asyncio.sleep(2 * attempt)
                await self.wait_for_host_connected(timeout_s=120.0, require_hello=require_hello)

        raise RuntimeError(f"Failed to enclave_sign after retries: {last_err}")

    async def enclave_sign_with_purpose(
        self,
        *,
        purpose: str,
        payload: bytes,
        preferred_algorithms=None,
        dp_keychain: Optional[bool] = None,
        cache_key: bool = True,
        challenge_id: Optional[str] = None,
    ) -> EnclaveSignResponse:
        key_info = await self.enclave_get_key_id(
            purpose=purpose,
            preferred_algorithms=preferred_algorithms or ["ES256"],
            dp_keychain=dp_keychain,
            cache=cache_key,
        )
        return await self.enclave_sign(
            key_id=key_info.key_id,
            payload=payload,
            dp_keychain=dp_keychain,
            challenge_id=challenge_id,
        )
