"""Pool training node that extends the default training with interactive wallet selection."""

from __future__ import annotations

import asyncio
from typing import Callable

from bittensor import Wallet
from common.models.api_models import MinerRegistrationResponse
from common.models.run_flags import RUN_FLAGS, update_run_flags
from common.utils.exceptions import SpecVersionException
from miner.utils.run_utils import get_miner_pool_run
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install
from loguru import logger

from miner.new_miner import Miner as BaseMiner
from miner import settings as miner_settings
from subnet.miner_api_client import RegisterMinerRequest

# from common.models.api_models import RegisterMinerRequest, MinerRegistrationResponse

from .wallet_utils import configure_payout_coldkey, determine_wallet_credentials
from .stats import StatsTracker
from .dashboard import MinerDashboard


_DEFAULT_POOL_COLDKEY = "iota"
_DEFAULT_POOL_HOTKEY = "iota_miner"


class Miner(BaseMiner):
    """Miner implementation that reuses `new_miner` behaviour with interactive wallet selection."""

    def __init__(
        self,
        wallet_name: str | None = None,
        wallet_hotkey: str | None = None,
        wallet: Wallet | None = None,
        run_btcli: Callable[[], str] | None = None,
        auto_start: bool = False,
        payout_coldkey: str | None = None,
        show_dashboard: bool = True,
        btcli_disabled: bool = False,
    ):
        detected_device = miner_settings.detect_device()
        device_type = "gpu" if detected_device == "cuda" else detected_device
        if miner_settings.DEVICE != detected_device:
            logger.info(
                "Auto-detected %s device; updating DEVICE from '%s' to '%s'.",
                device_type,
                miner_settings.DEVICE,
                detected_device,
            )
            miner_settings.set_device(detected_device)
        else:
            logger.info("Auto-detected %s device; DEVICE remains '%s'.", device_type, miner_settings.DEVICE)

        self.console = Console()
        install(console=self.console, show_locals=False, suppress=())

        self._btcli_disabled = btcli_disabled
        if self._btcli_disabled:
            logger.warning(
                "btcli integrations disabled. Loading pool wallet using local files only; "
                "interactive wallet creation and selection are unavailable."
            )

        if self._btcli_disabled:
            resolved_wallet_name, resolved_hotkey = self._determine_wallet_without_btcli(
                wallet_name=wallet_name,
                wallet_hotkey=wallet_hotkey,
                wallet=wallet,
            )
        else:
            resolved_wallet_name, resolved_hotkey = determine_wallet_credentials(
                wallet_name=wallet_name,
                wallet_hotkey=wallet_hotkey,
                wallet=wallet,
                run_btcli=run_btcli,
                console=self.console,
                auto_start=auto_start,
            )

        payout_coldkey_ss58 = configure_payout_coldkey(
            run_btcli=run_btcli,
            console=self.console,
            auto_start=auto_start,
            payout_override=payout_coldkey,
            btcli_disabled=self._btcli_disabled,
        )

        super().__init__(wallet_name=resolved_wallet_name, wallet_hotkey=resolved_hotkey, wallet=wallet)
        self._selected_payout_coldkey = payout_coldkey_ss58
        self._display_wallet_banner(resolved_wallet_name, resolved_hotkey, payout_coldkey_ss58)
        self.stats_tracker = StatsTracker()
        self.training_phase.attach_stats_tracker(self.stats_tracker)
        self._dashboard: MinerDashboard | None = (
            MinerDashboard(miner=self, tracker=self.stats_tracker, console=self.console) if show_dashboard else None
        )
        self._show_dashboard = show_dashboard

    def _determine_wallet_without_btcli(
        self,
        wallet_name: str | None,
        wallet_hotkey: str | None,
        wallet: Wallet | None,
    ) -> tuple[str, str]:
        """Resolve wallet names without invoking btcli helpers."""
        if wallet is not None:
            if wallet_name and wallet_hotkey:
                return wallet_name, wallet_hotkey
            raise RuntimeError(
                "When providing a Wallet instance, both wallet_name and wallet_hotkey must be set in --no-btcli mode."
            )

        if bool(wallet_name) ^ bool(wallet_hotkey):
            raise RuntimeError("Specify both --wallet and --hotkey when using --no-btcli.")

        resolved_coldkey = wallet_name or _DEFAULT_POOL_COLDKEY
        resolved_hotkey = wallet_hotkey or _DEFAULT_POOL_HOTKEY
        return resolved_coldkey, resolved_hotkey

    def _display_wallet_banner(
        self,
        wallet_name: str,
        wallet_hotkey: str,
        payout_coldkey_ss58: str | None,
    ) -> None:
        """Render a rich panel summarising the wallet selection."""
        payout_text = payout_coldkey_ss58 or "Creators (default)"
        payout_style = "bold cyan" if payout_coldkey_ss58 else "bold yellow"
        message = (
            f"Coldkey: [bold cyan]{wallet_name}[/]\n"
            f"Hotkey: [bold cyan]{wallet_hotkey}[/]\n"
            f"Payout coldkey: [{payout_style}]{payout_text}[/]\n"
            "[green]Wallet configuration complete. Initialising AI training...[/]"
        )
        self.console.print(
            Panel.fit(
                message,
                title="[bold green]AI Training Ready[/]",
                border_style="green",
            )
        )

    async def register_loop(self) -> tuple[dict, dict]:
        """
        Register the miner with the orchestrator, acquiring a layer during the process.
        If the miner is not registered, it will try to register every 60 seconds
        """
        while True:
            try:
                logger.info(f"🔄 Attempting to fetch run info for training node {self.hotkey[:8]}...")
                run_info_list = await self.miner_api_client.fetch_run_info_request()
                if not run_info_list:
                    raise Exception("Fatal Error: Could not fetch run info")

                best_run = get_miner_pool_run(run_info_list=run_info_list)
                logger.info(f"✅ Best run for training node {self.hotkey[:8]} is {best_run.run_id}")

                logger.info(
                    f"🔄 Attempting to register training node {self.hotkey[:8]} on run {best_run.run_id} with orchestrator..."
                )
                register_request = RegisterMinerRequest(run_id=best_run.run_id, register_as_metagraph_miner=False)
                response: MinerRegistrationResponse = await self.miner_api_client.register_miner_request(
                    register_miner_request=register_request
                )

                assigned_layer = int(response.layer)
                current_epoch = int(response.current_epoch)

                if response.layer is None:
                    raise Exception(
                        f"Training node {self.hotkey[:8]} registered with no layer assigned, this should not happen"
                    )

                if response.num_partitions is None:
                    raise Exception(f"Number of partitions is None for training node {self.hotkey[:8]}")

                logger.debug(f"Number of partitions for training node {self.hotkey[:8]}: {response.num_partitions}")

                self.model_manager.num_partitions = int(response.num_partitions)
                self.num_partitions = int(response.num_partitions)

                # TODO: clean these up
                self.layer = assigned_layer
                self.state_manager.layer = assigned_layer
                self.state_manager.training_epoch_when_registered = current_epoch
                self.state_manager.run_id = response.run_id
                self.run_id = response.run_id
                self.model_manager.epoch_on_registration = current_epoch

                update_run_flags(response.run_flags)
                _ = await self.miner_api_client.change_payout_coldkey_request(self._selected_payout_coldkey)

                logger.success(
                    f"✅ Training node {self.hotkey[:8]} registered successfully in layer {self.state_manager.layer} on training epoch {current_epoch}"
                )
                logger.debug(f"Run flags for training node {self.hotkey[:8]}: {RUN_FLAGS}")

                self.stats_tracker.reset()
                self.stats_tracker.set_layer(self.state_manager.layer)
                self.stats_tracker.set_remote_epoch(current_epoch)
                self.stats_tracker.set_run_id(response.run_id)

                return response.model_cfg.model_dump(), response.model_metadata.model_dump()

            except SpecVersionException as e:
                logger.error(f"Spec version mismatch: {e}")
                raise

            except Exception as e:
                logger.exception(f"Error registering training node: {e}")
                await asyncio.sleep(10)

    async def run_miner(self):
        """
        Run the miner with optional dashboard rendering.
        """
        if self._dashboard is not None:
            await self._dashboard.start()
        try:
            await super().run_miner()
        finally:
            if self._dashboard is not None:
                await self._dashboard.stop()
