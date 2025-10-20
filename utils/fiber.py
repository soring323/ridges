import utils.logger as logger

from fiber import Keypair



def validate_signed_timestamp(timestamp: int, signed_timestamp: str, hotkey: str) -> bool:
    """
    Checks if a signed timestamp is validly signed by the provided hotkey.
    """

    try:
        keypair = Keypair(ss58_address=hotkey)
        return keypair.verify(str(timestamp), bytes.fromhex(signed_timestamp))
    except Exception as e:
        logger.warning(f"Error in validate_signed_timestamp(timestamp={timestamp}, signed_timestamp={signed_timestamp}, hotkey={hotkey}): {e}")
        return False