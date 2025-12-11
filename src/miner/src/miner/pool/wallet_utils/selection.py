"""Interactive helpers for configuring pool miner wallets."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Tuple

from bittensor import Wallet
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


from substrateinterface import Keypair
from miner import settings as miner_settings
from common import settings as common_settings
from .btcli import parse_btcli_wallets
from .ui import clear_screen_with_banner, print_centered, prompt_ask_centered

WalletSelection = Tuple[str, str]

_DEFAULT_COLDKEY_NAME = "iota_miner_pool_coldkey"
_DEFAULT_HOTKEY_NAME = "iota_miner_pool_hotkey"
_MNEMONIC_PATTERN = re.compile(r"[a-z]+(?: [a-z]+){6,}")
_PAYOUT_PREFERENCE_FILE = Path.home() / ".bittensor" / "wallets" / ".pool_miner_payout.json"
CREATORS_PAYOUT_COLDKEY_PLACEHOLDER = common_settings.CREATORS_PAYOUT_COLDKEY_PLACEHOLDER


def _get_console(console: Console | None = None) -> Console:
    """Return the provided console or a module-level default."""
    return console or Console()


def _print_intro(console: Console) -> None:
    """Display the interactive setup header."""
    clear_screen_with_banner(console, subtitle="Interactive wallet setup")
    print_centered(console, "[bold]Let's configure your wallet access for the pool miner.[/]\n")


def _run_btcli_wallet_list(timeout: int = 30) -> str:
    """Run `btcli w list --json-out` and return stdout."""
    args = ["btcli", "w", "list", "--json-out"]
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "btcli executable not found. Install btcli or ensure it is on PATH before continuing."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("btcli command timed out while listing wallets.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        lowered = stderr.lower()
        if "unrecognized" in lowered or "unknown option" in lowered or "no such option" in lowered:
            raise RuntimeError(
                "This btcli version does not support --json-out. Upgrade btcli to a version that includes JSON output."
            ) from exc
        raise RuntimeError(f"btcli command failed (exit code {exc.returncode}): {stderr}") from exc

    return completed.stdout


def _prompt_from_list(
    items: Iterable[dict],
    header: str,
    formatter: Callable[[int, dict], str],
    prompt_text: str,
    console: Console,
):
    """Generic CLI prompt to select an item from a list."""
    items = list(items)
    if not items:
        raise ValueError("No items provided for selection.")

    error_message: str | None = None
    while True:
        clear_screen_with_banner(console, subtitle=header)
        table = Table(
            title=header,
            title_style="bold cyan",
            header_style="bold white",
            show_lines=True,
            expand=False,
        )
        table.add_column("#", justify="right", style="bold magenta")
        table.add_column("Details", style="white")
        for idx, item in enumerate(items, start=1):
            table.add_row(str(idx), formatter(idx, item))

        print_centered(console, table)
        if error_message:
            print_centered(console, f"[bold red]{error_message}[/]")
            error_message = None
        response = prompt_ask_centered(
            console,
            f"{prompt_text} [1-{len(items)}]",
        )
        try:
            option = int(response)
        except ValueError:
            error_message = "Please enter a number."
            continue

        if 1 <= option <= len(items):
            return items[option - 1]

        error_message = f"Please choose a number between 1 and {len(items)}."


def _wallet_exists(parsed_wallets: dict, coldkey_name: str, hotkey_name: str) -> bool:
    """Return True if the specified coldkey/hotkey pair exists in parsed wallets."""
    for coldkey in parsed_wallets.get("coldkeys") or []:
        if coldkey.get("name") != coldkey_name:
            continue
        for hotkey in coldkey.get("hotkeys") or []:
            if hotkey.get("name") == hotkey_name:
                return True
    return False


def _ensure_wallet_presence(
    run_btcli: Callable[[], str],
    console: Console,
    coldkey_name: str,
    hotkey_name: str,
    non_interactive: bool = False,
) -> None:
    """Guarantee a wallet exists; create it if missing."""
    parsed = parse_btcli_wallets(run_btcli())
    if not non_interactive:
        if _wallet_exists(parsed, coldkey_name, hotkey_name):
            clear_screen_with_banner(console, subtitle="Wallet Check")
            print_centered(
                console,
                Panel.fit(
                    f"Found existing wallet: [bold cyan]{coldkey_name}[/] / [bold cyan]{hotkey_name}[/]",
                    border_style="green",
                ),
            )
            return

        clear_screen_with_banner(console, subtitle="Wallet Check")
        print_centered(
            console,
            Panel.fit(
                "\n".join(
                    [
                        f"No wallet named [bold]{coldkey_name}[/] with hotkey [bold]{hotkey_name}[/] was found.",
                        "Creating it now using btcli...",
                    ]
                ),
                border_style="yellow",
            ),
        )
        create_new_wallet(coldkey_name, hotkey_name, console=console, non_interactive=non_interactive)
    else:
        if _wallet_exists(parsed, coldkey_name, hotkey_name):
            return
        create_new_wallet(coldkey_name, hotkey_name, console=console, non_interactive=non_interactive)


def create_new_wallet(
    coldkey_name: str,
    hotkey_name: str,
    console: Console | None = None,
    non_interactive: bool = False,
) -> WalletSelection:
    """Create a new wallet via btcli, store mnemonics, and return the names."""
    console = _get_console(console)
    try:
        completed = subprocess.run(
            [
                "btcli",
                "wallet",
                "create",
                "--name",
                coldkey_name,
                "--hotkey",
                hotkey_name,
                "--n-words",
                "21",
                "--no-use-password",
                "--wallet-path",
                "~/.bittensor/wallets",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "btcli executable not found. Install btcli or ensure it is on PATH before continuing."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("btcli command timed out while creating a wallet.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(f"btcli wallet creation failed (exit code {exc.returncode}): {stderr}") from exc

    stdout = completed.stdout.strip()
    mnemonics = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            _, after_colon = line.split(":", 1)
            candidate = after_colon.strip()
        else:
            candidate = line

        if candidate and _MNEMONIC_PATTERN.fullmatch(candidate):
            mnemonics.append(candidate)

    if len(mnemonics) < 2:
        raise RuntimeError(f"Unable to parse coldkey and hotkey mnemonics from btcli output. stdout: {stdout}")

    coldkey_mnemonic, hotkey_mnemonic = mnemonics[:2]

    wallet_root = Path.home() / ".bittensor" / "wallets" / coldkey_name
    wallet_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    mnemonic_path = wallet_root / f"{coldkey_name}_{hotkey_name}_mnemonics_{timestamp}.txt"
    mnemonic_path.write_text(
        (
            f"Coldkey name: {coldkey_name}\n"
            f"Coldkey mnemonic: {coldkey_mnemonic}\n\n"
            f"Hotkey name: {hotkey_name}\n"
            f"Hotkey mnemonic: {hotkey_mnemonic}\n"
        ),
        encoding="utf-8",
    )
    try:
        mnemonic_path.chmod(0o600)
    except OSError:
        pass

    if not non_interactive:
        print_centered(
            console,
            Panel.fit(
                f"Mnemonics saved to [bold]{mnemonic_path}[/]. Store this file securely and preferably offline.",
                border_style="yellow",
            ),
        )
        print_centered(
            console, "[bold]Confirm that you understand where to find your mnemonics: [bold green]y[/]/[bold red]n[/]"
        )
        confirmation = console.input("")
        if confirmation.lower() != "y":
            print_centered(console, "[bold red]Exiting miner setup at your request.[/]")
            raise SystemExit(0)

    return coldkey_name, hotkey_name


def _select_local_coldkey(run_btcli: Callable[[], str], console: Console) -> str:
    """Return the ss58 address of a coldkey chosen from the local machine."""
    parsed = parse_btcli_wallets(run_btcli())
    coldkeys = parsed.get("coldkeys") or []
    if not coldkeys:
        raise RuntimeError("No coldkeys found on this machine. Create one first.")

    selection = _prompt_from_list(
        coldkeys,
        header="Available Coldkeys",
        formatter=lambda idx, coldkey: f"{coldkey['name']} ({coldkey['ss58_address']})",
        prompt_text="Select coldkey",
        console=console,
    )
    address = selection.get("ss58_address")
    if not address:
        raise RuntimeError("Selected coldkey does not contain an ss58 address.")

    clear_screen_with_banner(console, subtitle="Coldkey Selected")
    print_centered(
        console,
        Panel.fit(
            f"Selected coldkey [bold]{selection['name']}[/] → [cyan]{address}[/]",
            border_style="green",
        ),
    )
    return address


def _prompt_manual_payout_coldkey(console: Console) -> str:
    """Prompt the operator to enter a payout coldkey address manually."""
    error_message: str | None = None
    while True:
        clear_screen_with_banner(console, subtitle="Manual payout coldkey")
        if error_message:
            print_centered(console, f"[bold red]{error_message}[/]")
        raw_address = prompt_ask_centered(console, "Enter payout coldkey ss58 address")
        if raw_address:
            clear_screen_with_banner(console, subtitle="Manual payout coldkey")
            print_centered(
                console,
                Panel.fit(
                    f"Using payout coldkey address [cyan]{raw_address}[/]",
                    border_style="green",
                ),
            )
            return raw_address
        error_message = "Address cannot be empty."


def _prompt_for_payout_coldkey(
    run_btcli: Callable[[], str] | None,
    console: Console,
    allow_local_selection: bool = True,
) -> str:
    """Ask the operator how to determine the payout coldkey and return the address or None."""
    while True:
        clear_screen_with_banner(console, subtitle="Payout coldkey")
        options: list[str] = ["How you want to specify the payout coldkey"]
        choices: list[str]
        if allow_local_selection:
            options.extend(
                [
                    "[1] Choose one from this machine",
                    "[2] Paste your ss58 address",
                    (
                        "[3] Don't specify a payout coldkey (proceeds go to the creators) "
                        f"[{CREATORS_PAYOUT_COLDKEY_PLACEHOLDER}]"
                    ),
                ]
            )
            choices = ["1", "2", "3"]
        else:
            options.extend(
                [
                    "[yellow]Local wallet selection is disabled because btcli is not available.[/]",
                    "[1] Paste your ss58 address",
                    (
                        "[2] Don't specify a payout coldkey (proceeds go to the creators) "
                        f"[{CREATORS_PAYOUT_COLDKEY_PLACEHOLDER}]"
                    ),
                ]
            )
            choices = ["1", "2"]

        print_centered(
            console,
            Panel.fit("\n".join(options), border_style="cyan"),
        )
        choice = prompt_ask_centered(
            console,
            "Select option",
            choices=choices,
            default="1",
        )

        if allow_local_selection:
            if choice == "1":
                if run_btcli is None:
                    raise RuntimeError("btcli integration is required for local coldkey selection.")
                payout_coldkey = _select_local_coldkey(run_btcli, console)
                return payout_coldkey
            if choice == "2":
                payout_coldkey = _prompt_manual_payout_coldkey(console)
                return payout_coldkey
        else:
            if choice == "1":
                payout_coldkey = _prompt_manual_payout_coldkey(console)
                return payout_coldkey

        clear_screen_with_banner(console, subtitle="Payout coldkey")
        print_centered(
            console,
            Panel.fit(
                (
                    "No payout coldkey specified. Pool rewards will be routed to the pool creators "
                    f"[{CREATORS_PAYOUT_COLDKEY_PLACEHOLDER}]."
                ),
                border_style="yellow",
            ),
        )
        return CREATORS_PAYOUT_COLDKEY_PLACEHOLDER


def is_valid_ss58(address: str, ss58_format: int = None) -> bool:
    """
    Check if a string is a valid SS58 address.

    Args:
        address: The SS58 address string to validate
        ss58_format: Optional SS58 format number (e.g., 42 for generic Substrate)
                    If None, validates against any SS58 format

    Returns:
        True if valid, False otherwise
    """
    try:
        # This will raise an exception if the address is invalid
        Keypair(ss58_address=address)
        return True
    except Exception:
        return False


# Usage examples
print(is_valid_ss58("5DWq6LBGKa7GxbpEJkAjuW3q94iDYAp9L117uLncFwyXNg8e"))  # Valid
print(is_valid_ss58("invalid_address"))  # False

# Check against specific SS58 format (e.g., Polkadot = 0, Kusama = 2, Generic = 42)
print(is_valid_ss58(r"5DWq6LBGKa7GxbpEJkAjuW3q94iDYAp9L117uLncFwyXNg8e", ss58_format=42))


def _load_saved_payout_coldkey() -> tuple[bool, str | None]:
    """Retrieve the stored payout coldkey selection."""
    try:
        raw = _PAYOUT_PREFERENCE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False, None
    except OSError:
        return False, None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, None

    if "payout_coldkey" in data:
        saved_value = str(data.get("payout_coldkey"))
        if isinstance(saved_value, str) and is_valid_ss58(saved_value):
            return True, saved_value
        print(f"Attempted to load invalid payout coldkey: {saved_value}")
        return False, None
    return False, None


def _save_payout_coldkey(payout_coldkey: str) -> None:
    """Persist the payout coldkey selection so it can be reused automatically."""
    try:
        _PAYOUT_PREFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        serialized_value = payout_coldkey.strip()
        _PAYOUT_PREFERENCE_FILE.write_text(json.dumps({"payout_coldkey": serialized_value}), encoding="utf-8")
    except OSError:
        # Persisting the preference improves UX but is not critical.
        pass


def configure_payout_coldkey(
    run_btcli: Callable[[], str] | None = None,
    console: Console | None = None,
    auto_start: bool = False,
    payout_override: str | None = None,
    btcli_disabled: bool = False,
) -> str:
    """
    Prompt for payout coldkey selection and present the main action menu.

    Returns the ss58 address to use for payouts, or None if default routing should be used.
    May raise SystemExit if the operator chooses to quit.
    """
    allow_local_selection = not btcli_disabled
    run_btcli_callable: Callable[[], str] | None = None
    if allow_local_selection:
        run_btcli_callable = run_btcli or _run_btcli_wallet_list

    override_specified = payout_override is not None
    if override_specified:
        normalized_override = payout_override.strip()
        if not normalized_override or normalized_override.lower() in {"creators", "default"}:
            normalized_override = CREATORS_PAYOUT_COLDKEY_PLACEHOLDER

    has_saved, payout_coldkey = _load_saved_payout_coldkey()

    if override_specified:
        payout_coldkey = normalized_override
        _save_payout_coldkey(payout_coldkey)
        has_saved = True

    if auto_start:
        if not payout_coldkey:
            payout_coldkey = CREATORS_PAYOUT_COLDKEY_PLACEHOLDER
        if not has_saved:
            _save_payout_coldkey(payout_coldkey)
        return payout_coldkey

    console = _get_console(console)

    if has_saved:
        message = (
            f"Restored saved payout coldkey: [cyan]{payout_coldkey}[/]"
            if payout_coldkey
            else (
                "Restored payout setting: [yellow]Creators (default routing) "
                f"[{CREATORS_PAYOUT_COLDKEY_PLACEHOLDER}][/]"
            )
        )
        clear_screen_with_banner(console, subtitle="Payout coldkey")
        print_centered(
            console,
            Panel.fit(message, border_style="green"),
        )
    else:
        payout_coldkey = _prompt_for_payout_coldkey(
            run_btcli_callable,
            console,
            allow_local_selection=allow_local_selection,
        )
        _save_payout_coldkey(payout_coldkey)

    while True:
        clear_screen_with_banner(console, subtitle="Main Menu")

        device = miner_settings.DEVICE or miner_settings.detect_device()
        if device == "cuda":
            device_display = "GPU (cuda)"
        elif device == "mps":
            device_display = "MPS"
        else:
            device_display = device.upper()

        current_payout_display = payout_coldkey or f"Creators (default: {CREATORS_PAYOUT_COLDKEY_PLACEHOLDER})"

        menu_panel = Panel.fit(
            "\n".join(
                [
                    "[bold]Pool Miner Menu[/]",
                    f"Current payout coldkey: [cyan]{current_payout_display}[/]",
                    f"Detected device: [cyan]{device_display}[/]",
                    "",
                    "[1] Start miner",
                    "[2] Change payout coldkey",
                    "[3] Quit",
                ]
            ),
            border_style="magenta",
        )
        print_centered(console, menu_panel)

        selection = prompt_ask_centered(
            console,
            "Select option",
            choices=["1", "2", "3"],
            default="1",
        )

        if selection == "1":
            if not payout_coldkey:
                payout_coldkey = CREATORS_PAYOUT_COLDKEY_PLACEHOLDER
            _save_payout_coldkey(payout_coldkey)
            return payout_coldkey
        if selection == "2":
            payout_coldkey = _prompt_for_payout_coldkey(
                run_btcli_callable,
                console,
                allow_local_selection=allow_local_selection,
            )
            _save_payout_coldkey(payout_coldkey)
            continue
        clear_screen_with_banner(console, subtitle="Goodbye")
        print_centered(console, "[yellow]Exiting miner setup at your request.[/]")
        raise SystemExit(0)


def determine_wallet_credentials(
    wallet_name: str | None,
    wallet_hotkey: str | None,
    wallet: Wallet | None,
    run_btcli: Callable[[], str] | None = None,
    console: Console | None = None,
    auto_start: bool = False,
) -> WalletSelection:
    """Determine which wallet credentials to use for the miner."""
    run_btcli = run_btcli or _run_btcli_wallet_list
    console = _get_console(console)

    if wallet is not None:
        if wallet_name and wallet_hotkey:
            return wallet_name, wallet_hotkey
        raise RuntimeError("Wallet instance provided without explicit wallet_name and wallet_hotkey strings.")

    if bool(wallet_name) ^ bool(wallet_hotkey):
        raise RuntimeError("Both wallet_name and wallet_hotkey must be provided together.")

    if wallet_name and wallet_hotkey:
        _ensure_wallet_presence(
            run_btcli=run_btcli,
            console=console,
            coldkey_name=wallet_name,
            hotkey_name=wallet_hotkey,
            non_interactive=auto_start,
        )
        return wallet_name, wallet_hotkey

    if auto_start:
        _ensure_wallet_presence(
            run_btcli=run_btcli,
            console=console,
            coldkey_name=_DEFAULT_COLDKEY_NAME,
            hotkey_name=_DEFAULT_HOTKEY_NAME,
            non_interactive=True,
        )
        return _DEFAULT_COLDKEY_NAME, _DEFAULT_HOTKEY_NAME

    _print_intro(console)
    _ensure_wallet_presence(
        run_btcli=run_btcli,
        console=console,
        coldkey_name=_DEFAULT_COLDKEY_NAME,
        hotkey_name=_DEFAULT_HOTKEY_NAME,
    )

    clear_screen_with_banner(console, subtitle="Wallet selection")
    print_centered(
        console,
        Panel.fit(
            f"Using wallet [bold cyan]{_DEFAULT_COLDKEY_NAME}[/] with hotkey [bold cyan]{_DEFAULT_HOTKEY_NAME}[/].",
            border_style="green",
        ),
    )
    return _DEFAULT_COLDKEY_NAME, _DEFAULT_HOTKEY_NAME
