import time
import asyncio
import utils.logger as logger



DEBUG_LOCKS = {
    "waiting": [],
    "locked": [],
    "slow": []
}

DEBUG_LOCKS_LOCK = asyncio.Lock()



class DebugLock:
    def __init__(self, lock: asyncio.Lock, label: str):
        self.lock = lock
        self.label = label
        self.waiting_at = None
        self.acquired_at = None

    async def __aenter__(self):
        self.waiting_at = time.time()
        logger.info(f"[DebugLock] {self.label}: Trying to acquire lock...")
        waiting_entry = {
            "label": self.label,
            "waiting_at": self.waiting_at
        }
        async with DEBUG_LOCKS_LOCK:
            DEBUG_LOCKS["waiting"].append(waiting_entry)
        await self.lock.acquire()
        self.acquired_at = time.time()
        acquired_entry = {
            "label": self.label,
            "waiting_at": self.waiting_at,
            "acquired_at": self.acquired_at
        }
        elapsed = self.acquired_at - self.waiting_at
        async with DEBUG_LOCKS_LOCK:
            DEBUG_LOCKS["waiting"] = [x for x in DEBUG_LOCKS["waiting"] if not (x["label"] == self.label and x["waiting_at"] == self.waiting_at)]
            DEBUG_LOCKS["locked"].append(acquired_entry)
            if elapsed > 5:
                DEBUG_LOCKS["slow"].append(acquired_entry)
        logger.info(f"[DebugLock] {self.label}: Lock acquired after {elapsed:.2f} seconds")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.lock.release()
        async with DEBUG_LOCKS_LOCK:
            DEBUG_LOCKS["locked"] = [x for x in DEBUG_LOCKS["locked"] if not (x["label"] == self.label and x["waiting_at"] == self.waiting_at)]
        logger.info(f"[DebugLock] {self.label}: Lock released")



def get_debug_locks_info():
    now = time.time()

    waiting_info = []
    for entry in DEBUG_LOCKS["waiting"]:
        seconds_waiting = now - entry["waiting_at"]
        waiting_info.append([entry["label"], f"{seconds_waiting:.2f} s"])

    locked_info = []
    for entry in DEBUG_LOCKS["locked"]:
        seconds_locked = now - entry["acquired_at"]
        locked_info.append([entry["label"], f"{seconds_locked:.2f} s"])

    slow_info = []
    for entry in DEBUG_LOCKS["slow"]:
        seconds_to_lock = entry["acquired_at"] - entry["waiting_at"]
        slow_info.append([entry["label"], f"{seconds_to_lock:.2f} s"])

    return {
        "waiting": waiting_info,
        "locked": locked_info,
        "slow": slow_info
    }