import os
import re
import utils.logger as logger

from dotenv import load_dotenv
from fiber.chain.chain_utils import load_hotkey_keypair



# Load everything from .env
load_dotenv()



# Bittensor configuration
NETUID = os.getenv("NETUID")
if not NETUID:
    logger.fatal("NETUID is not set in .env")
NETUID = int(NETUID)

SUBTENSOR_ADDRESS = os.getenv("SUBTENSOR_ADDRESS")
if not SUBTENSOR_ADDRESS:
    logger.fatal("SUBTENSOR_ADDRESS is not set in .env")

SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK")
if not SUBTENSOR_NETWORK:
    logger.fatal("SUBTENSOR_NETWORK is not set in .env")



# Load the mode
MODE = os.getenv("MODE")
if not MODE:
    logger.fatal("MODE is not set in .env")

if MODE != "screener" and MODE != "validator":
    logger.fatal("MODE must be either 'screener' or 'validator'")



if MODE == "validator":
    # Load wallet name and hotkey name
    VALIDATOR_WALLET_NAME = os.getenv("VALIDATOR_WALLET_NAME")
    if not VALIDATOR_WALLET_NAME:
        logger.fatal("VALIDATOR_WALLET_NAME is not set in .env")

    VALIDATOR_HOTKEY_NAME = os.getenv("VALIDATOR_HOTKEY_NAME")
    if not VALIDATOR_HOTKEY_NAME:
        logger.fatal("VALIDATOR_HOTKEY_NAME is not set in .env")

    # Load the hotkey
    try:
        VALIDATOR_HOTKEY = load_hotkey_keypair(VALIDATOR_WALLET_NAME, VALIDATOR_HOTKEY_NAME)
    except Exception as e:
        logger.fatal(f"Error loading hotkey: {e}")



elif MODE == "screener":
    # Load the screener name
    SCREENER_NAME = os.getenv("SCREENER_NAME")
    if not SCREENER_NAME:
        logger.fatal("SCREENER_NAME is not set in .env")

    # Ensure that the screener name is in the format screener-CLASS-NUM
    if not re.match(r"screener-\d-\d+", SCREENER_NAME):
        logger.fatal("SCREENER_NAME must be in the format screener-CLASS-NUM")

    # Ensure that CLASS is either 1 or 2
    screener_class = SCREENER_NAME.split("-")[1]
    if screener_class != "1" and screener_class != "2":
        logger.fatal("SCREENER_NAME must be in the format screener-CLASS-NUM where CLASS is 1 or 2")

    # Load the screener password
    SCREENER_PASSWORD = os.getenv("SCREENER_PASSWORD")
    if not SCREENER_PASSWORD:
        logger.fatal("SCREENER_PASSWORD is not set in .env")



# Load URLs (and strip trailing slashes)
RIDGES_PLATFORM_URL = os.getenv("RIDGES_PLATFORM_URL")
if not RIDGES_PLATFORM_URL:
    logger.fatal("RIDGES_PLATFORM_URL is not set in .env")

RIDGES_PLATFORM_URL = RIDGES_PLATFORM_URL.rstrip("/")

RIDGES_INFERENCE_GATEWAY_URL = os.getenv("RIDGES_INFERENCE_GATEWAY_URL")
if not RIDGES_INFERENCE_GATEWAY_URL:
    logger.fatal("RIDGES_INFERENCE_GATEWAY_URL is not set in .env")

RIDGES_INFERENCE_GATEWAY_URL = RIDGES_INFERENCE_GATEWAY_URL.rstrip("/")



# Load the time to wait between sending heartbeats
SEND_HEARTBEAT_INTERVAL_SECONDS = os.getenv("SEND_HEARTBEAT_INTERVAL_SECONDS")
if not SEND_HEARTBEAT_INTERVAL_SECONDS:
    logger.fatal("SEND_HEARTBEAT_INTERVAL_SECONDS is not set in .env")
SEND_HEARTBEAT_INTERVAL_SECONDS = int(SEND_HEARTBEAT_INTERVAL_SECONDS) 

# Load the time to wait between setting weights
SET_WEIGHTS_INTERVAL_SECONDS = os.getenv("SET_WEIGHTS_INTERVAL_SECONDS")
if not SET_WEIGHTS_INTERVAL_SECONDS:
    logger.fatal("SET_WEIGHTS_INTERVAL_SECONDS is not set in .env")
SET_WEIGHTS_INTERVAL_SECONDS = int(SET_WEIGHTS_INTERVAL_SECONDS) 

# Load the time to wait between requesting a new evaluation
REQUEST_EVALUATION_INTERVAL_SECONDS = os.getenv("REQUEST_EVALUATION_INTERVAL_SECONDS")
if not REQUEST_EVALUATION_INTERVAL_SECONDS:
    logger.fatal("REQUEST_EVALUATION_INTERVAL_SECONDS is not set in .env")
REQUEST_EVALUATION_INTERVAL_SECONDS = int(REQUEST_EVALUATION_INTERVAL_SECONDS) 



# Load the simulated evaluation runs configuration
SIMULATE_EVALUATION_RUNS = os.getenv("SIMULATE_EVALUATION_RUNS")
if not SIMULATE_EVALUATION_RUNS:
    logger.fatal("SIMULATE_EVALUATION_RUNS is not set in .env")
SIMULATE_EVALUATION_RUNS = SIMULATE_EVALUATION_RUNS.lower() == "true"

SIMULATE_EVALUATION_RUN_MAX_TIME_PER_STAGE_SECONDS = os.getenv("SIMULATE_EVALUATION_RUN_MAX_TIME_PER_STAGE_SECONDS")
if not SIMULATE_EVALUATION_RUN_MAX_TIME_PER_STAGE_SECONDS:
    logger.fatal("SIMULATE_EVALUATION_RUN_MAX_TIME_PER_STAGE_SECONDS is not set in .env")
SIMULATE_EVALUATION_RUN_MAX_TIME_PER_STAGE_SECONDS = int(SIMULATE_EVALUATION_RUN_MAX_TIME_PER_STAGE_SECONDS)

INCLUDE_SOLUTIONS = os.getenv("INCLUDE_SOLUTIONS")
if not INCLUDE_SOLUTIONS:
    logger.fatal("INCLUDE_SOLUTIONS is not set in .env")
INCLUDE_SOLUTIONS = INCLUDE_SOLUTIONS.lower() == "true"



# Load the update automatically configuration
UPDATE_AUTOMATICALLY = os.getenv("UPDATE_AUTOMATICALLY")
if not UPDATE_AUTOMATICALLY:
    logger.fatal("UPDATE_AUTOMATICALLY is not set in .env")
UPDATE_AUTOMATICALLY = UPDATE_AUTOMATICALLY.lower() == "true"



# Print out the configuration
logger.info("=== Validator Configuration ===")
logger.info(f"Network ID: {NETUID}")
logger.info(f"Subtensor Address: {SUBTENSOR_ADDRESS}")
logger.info(f"Subtensor Network: {SUBTENSOR_NETWORK}")
logger.info("-------------------------------")
logger.info(f"Mode: {MODE}")
if MODE == "validator":
    logger.info(f"Validator Wallet Name: {VALIDATOR_WALLET_NAME}")
    logger.info(f"Validator Hotkey Name: {VALIDATOR_HOTKEY_NAME}")
    logger.info(f"Validator Hotkey: {VALIDATOR_HOTKEY.ss58_address}")
elif MODE == "screener":
    logger.info(f"Screener Name: {SCREENER_NAME}")
logger.info("-------------------------------")
logger.info(f"Ridges Platform URL: {RIDGES_PLATFORM_URL}")
logger.info(f"Ridges Inference Gateway URL: {RIDGES_INFERENCE_GATEWAY_URL}")
logger.info("-------------------------------")
logger.info(f"Send Heartbeat Interval: {SEND_HEARTBEAT_INTERVAL_SECONDS} second(s)")
logger.info(f"Set Weights Interval: {SET_WEIGHTS_INTERVAL_SECONDS} second(s)")
logger.info(f"Request Evaluation Interval: {REQUEST_EVALUATION_INTERVAL_SECONDS} second(s)")
logger.info("-------------------------------")
if SIMULATE_EVALUATION_RUNS:
    logger.warning("Simulating Evaluation Runs!")
else:
    if INCLUDE_SOLUTIONS:
        logger.warning("Including Solutions!")
    else:
        logger.info("Not Including Solutions")
logger.info("-------------------------------")
if UPDATE_AUTOMATICALLY:
    logger.info("Updating Automatically")
else:
    logger.warning("Not Updating Automatically")
logger.info("===============================")