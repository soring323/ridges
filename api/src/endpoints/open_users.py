from typing import Optional
from bittensor_wallet.utils import is_valid_ss58_address
from fastapi import APIRouter, HTTPException
import secrets
import string
from datetime import datetime
import os
from dotenv import load_dotenv

from api.src.backend.queries.open_users import get_open_user, create_open_user, get_open_user_by_email, update_open_user_bittensor_hotkey as db_update_open_user_bittensor_hotkey, get_open_user_bittensor_hotkey as db_get_open_user_bittensor_hotkey, get_emission_dispersed_to_open_user as db_get_emission_dispersed_to_open_user, get_treasury_transactions_for_open_user as db_get_treasury_transactions_for_open_user, get_open_agent_periods_on_top as db_get_open_agent_periods_on_top, get_periods_on_top_map as db_get_periods_on_top_map, get_total_payouts_by_agent_ids as db_get_total_payouts_by_agent_ids
from api.src.backend.queries.scores import get_treasury_hotkeys as db_get_treasury_hotkeys
from api.src.backend.queries.agents import get_agent_by_agent_id as db_get_agent_by_agent_id
from api.src.backend.entities import OpenUser, OpenUserSignInRequest
import utils.logger as logger

load_dotenv()

router = APIRouter()

open_user_password = os.getenv("OPEN_USER_PASSWORD")

@router.post("/sign-in", tags=["open-users"])
async def open_user_sign_in(request: OpenUserSignInRequest):
    auth0_user_id = request.auth0_user_id
    email = request.email
    name = request.name
    password = request.password

    if password != open_user_password:
        logger.warning(f"Someone tried to sign in with an invalid password. auth0_user_id: {auth0_user_id}, email: {email}, name: {name}, password: {password}")
        raise HTTPException(status_code=401, detail="Invalid sign in password. Fuck you.")

    logger.info(f"Open user sign in process beginning for: {auth0_user_id}, {email}, {name}")

    existing_user = await get_open_user(auth0_user_id)

    if existing_user:
        logger.info(f"Open user {existing_user.open_hotkey} signed in successfully")
        return {"success": True, "new_user": False, "message": "User exists", "user": existing_user}
    
    new_user = OpenUser(
        open_hotkey="open-" + ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(47)),
        auth0_user_id=auth0_user_id,
        email=email,
        name=name,
        registered_at=datetime.now()
    )

    try:
        await create_open_user(new_user)
        await db_update_open_user_bittensor_hotkey(new_user.open_hotkey, None)
    except Exception as e:
        logger.error(f"Error creating open user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later and message us on Discord if the problem persists.")
    
    logger.info(f"Open user created: {new_user.open_hotkey}")
    return {"success": True, "new_user": True, "message": "User successfully created", "user": new_user}

@router.get("/get-user-by-email", tags=["open-users"])
async def get_user_by_email(email: str, password: str):
    if password != open_user_password:
        logger.warning(f"Someone tried to get user by email with an invalid password. email: {email}, password: {password}")
        raise HTTPException(status_code=401, detail="Invalid password. Fuck you.")

    try:
        user = await get_open_user_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        bittensor_hotkey = await db_get_open_user_bittensor_hotkey(user.open_hotkey)
        user.bittensor_hotkey = bittensor_hotkey
        
        return {"success": True, "user": user}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user by email {email}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later and message us on Discord if the problem persists.")

@router.post("/update-bittensor-hotkey", tags=["open-users"])
async def update_bittensor_hotkey(open_hotkey: str, password: str, bittensor_hotkey: Optional[str] = None):
    if password != open_user_password:
        logger.warning(f"Someone tried to update bittensor hotkey with an invalid password. open_hotkey: {open_hotkey}, bittensor_hotkey: {bittensor_hotkey}, password: {password}")
        raise HTTPException(status_code=401, detail="Invalid password. Fuck you.")

    if bittensor_hotkey and not is_valid_ss58_address(bittensor_hotkey):
        raise HTTPException(status_code=400, detail="Invalid bittensor hotkey. Please provide a valid SS58 address.")
    try:
        await db_update_open_user_bittensor_hotkey(open_hotkey, bittensor_hotkey)
        return {"success": True, "message": "Bittensor hotkey updated"}
    except Exception as e:
        logger.error(f"Error updating bittensor hotkey for open user {open_hotkey}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later and message us on Discord if the problem persists.")

@router.get("/get-treasury-transactions-for-open-user", tags=["open-users"])
async def get_treasury_transactions_for_open_user(open_hotkey: str, password: str):
    if password != open_user_password:
        logger.warning(f"Someone tried to get treasury transactions for open user with an invalid password. open_hotkey: {open_hotkey}, password: {password}")
        raise HTTPException(status_code=401, detail="Invalid password. Fuck you.")
    
    try:
        treasury_transactions = await db_get_treasury_transactions_for_open_user(open_hotkey)
        return {"success": True, "treasury_transactions": treasury_transactions, "open_hotkey": open_hotkey}
    except Exception as e:
        logger.error(f"Error getting treasury transactions for open user {open_hotkey}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later and message us on Discord if the problem persists.")

@router.get("/get-pending-emission-for-open-user", tags=["open-users"])
async def get_pending_emission_for_open_user(open_hotkey: str, password: str):
    if password != open_user_password:
        logger.warning(f"Someone tried to get pending emission for open user with an invalid password. open_hotkey: {open_hotkey}, password: {password}")
        raise HTTPException(status_code=401, detail="Invalid password. Fuck you.")
    
    try:
        periods_on_top = await db_get_open_agent_periods_on_top(miner_hotkey=open_hotkey)
        if not periods_on_top:
            return {"success": True, "pending_emission": 0, "open_hotkey": open_hotkey}
        
        treasury_hotkeys = await db_get_treasury_hotkeys()
        gross_emission = await internal_tools.get_emission_alpha_for_hotkeys_during_periods(miner_hotkeys=treasury_hotkeys, periods=periods_on_top)
        emission_dispersed = await db_get_emission_dispersed_to_open_user(open_hotkey)
        return {"success": True, "pending_emission": gross_emission - emission_dispersed, "open_hotkey": open_hotkey}
    except Exception as e:
        logger.error(f"Error getting pending emission for open user {open_hotkey}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later and message us on Discord if the problem persists.")

@router.get("/get-all-pending-payouts", tags=["open-users"])
async def get_all_pending_payouts(password: str):
    if password != open_user_password:
        logger.warning(f"Someone tried to get all pending payouts with an invalid password. password: {password}")
        raise HTTPException(status_code=401, detail="Invalid password. Fuck you.")
    
    try:
        periods_on_top_map = await db_get_periods_on_top_map()
        treasury_hotkeys = await db_get_treasury_hotkeys()
        gross_emissions = await internal_tools.get_emission_alpha_map_for_hotkeys_during_periods(miner_hotkeys=treasury_hotkeys, periods_map=periods_on_top_map)
        payouts = await db_get_total_payouts_by_agent_ids(agent_ids=list(gross_emissions.keys()))

        hydrated_payouts = {}
        for agent_id in gross_emissions.keys():
            agent = await db_get_agent_by_agent_id(agent_id)
            agent_name = agent.name if agent else None
            agent_hotkey = agent.miner_hotkey if agent else None
            bittensor_hotkey = await db_get_open_user_bittensor_hotkey(agent_hotkey) if agent_hotkey else None
            periods = periods_on_top_map.get(agent_id, [])
            periods_on_top = [(str(start), str(end)) for start, end in periods]

            realized = payouts.get(agent_id, 0)
            gross = gross_emissions.get(agent_id, 0)
            pending = gross - realized

            hydrated_payouts[agent_id] = {
                "pending_payout": pending,
                "realized_payout": realized,
                "gross_payout": gross,
                "agent_name": agent_name,
                "agent_hotkey": agent_hotkey,
                "periods_on_top": periods_on_top,
                "bittensor_hotkey": bittensor_hotkey,
            }

        return {"success": True, "pending_payouts": hydrated_payouts}
    except Exception as e:
        logger.error(f"Error getting all pending payouts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later and message us on Discord if the problem persists.")
