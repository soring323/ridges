from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional
from queries.statistics import score_improvement_24_hrs, agents_created_24_hrs, top_score
import utils.logger as logger
from dotenv import load_dotenv

from api.src.utils.auth import verify_request_public
from api.src.backend.queries.statistics import get_top_scores_over_last_week as db_get_top_scores_over_last_week


load_dotenv()

router = APIRouter()


import uuid
import asyncio
from queries.agent import get_agents_in_queue, get_top_agents, get_agent_by_id, get_latest_agent_for_hotkey
from queries.evaluation import get_evaluations_for_agent_id
from queries.evaluation_run import get_all_evaluation_runs_in_evaluation_id
from models.evaluation import Evaluation, EvaluationWithRuns
from models.evaluation_set import EvaluationSetGroup
from models.agent import Agent, AgentScored, AgentStatus
from utils.s3 import download_text_file_from_s3
from api.src.endpoints.validator import get_all_connected_validator_ip_addresses

async def queue(
    stage: str
) -> list[Agent]:
    """
    Gets agents presently in queue in order.
    
    Args:
        stage: screener_1, screener_2, or validator.

    Returns:
        A list of Agent objects, sorted by their position in queue
    """
    return await get_agents_in_queue(EvaluationSetGroup(stage))

async def top_agents(
    number_of_agents: int = 5,
    page: int = 1
) -> list[AgentScored]:
    """
    Returns the top agents for the latest problem set. All agents, including ones that have not been approved.
    """
    return await get_top_agents(
        number_of_agents=number_of_agents,
        page=page
    )

async def agent_by_id(agent_id: str) -> Agent:
    agent = await get_agent_by_id(agent_id=uuid.UUID(agent_id))
    
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )

    return agent

async def agent_by_hotkey(miner_hotkey: str) -> Agent:
    """
    Returns the latest agent submitted by a hotkey
    """
    agent = await get_latest_agent_for_hotkey(miner_hotkey=miner_hotkey)
    
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )

    return agent

# TODO ADAM: optimize that
async def evaluations_for_agent(agent_id: str) -> list[EvaluationWithRuns]:
    evaluations: list[Evaluation] = await get_evaluations_for_agent_id(agent_id=uuid.UUID(agent_id))
    
    runs_per_eval = await asyncio.gather(
        *[get_all_evaluation_runs_in_evaluation_id(evaluation_id=e.evaluation_id) for e in evaluations]
    )

    return [
        EvaluationWithRuns(evaluation=e, runs=runs)
        for e, runs in zip(evaluations, runs_per_eval)
    ]

async def get_agent_code(agent_id: str, request: Request):
    agent_version = await get_agent_by_id(agent_id=agent_id)
    
    if not agent_version:
        logger.info(f"File for agent version {agent_id} was requested but not found in our database")
        raise HTTPException(
            status_code=404, 
            detail="The requested agent version was not found. Are you sure you have the correct version ID?"
        )
    
    if agent_version.status in [AgentStatus.screening_1, AgentStatus.screening_2, AgentStatus.evaluating]:
        client_ip = request.client.host
        
        connected_validator_ips = get_all_connected_validator_ip_addresses()

        if client_ip not in connected_validator_ips:
            logger.warning(f"Unauthorized IP {client_ip} attempted to access agent code for version {agent_id}")
            raise HTTPException(
                status_code=403,
                detail="Access denied: IP not authorized"
            )
    
    try:
        text = await download_text_file_from_s3(f"{agent_id}/agent.py")
    except Exception as e:
        logger.error(f"Error retrieving agent version code from S3 for version {agent_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving agent version code. Please try again later."
        )
    
    return text

async def top_scores_over_time():
    """Gets agent scores over time for charting"""
    return await db_get_top_scores_over_last_week()

async def network_statistics():
    """
    Gets network statistics for the dashboard
    """
    score_improvement, agents_created, top_score_value = await asyncio.gather(
        score_improvement_24_hrs(),
        agents_created_24_hrs(),
        top_score()
    )
    return {
        "score_improvement_24_hrs": score_improvement,
        "agents_created_24_hrs": agents_created,
        "top_score": top_score_value
    }

router = APIRouter()

routes = [
    ("/queue", queue),
    ("/top-agents", top_agents),
    ("/agent-by-id", agent_by_id),
    ("/agent-by-hotkey", agent_by_hotkey),
    ("/evaluations-for-agent", evaluations_for_agent),
    ("/agent-version-file", get_agent_code),
    ("/top-scores-over-time", top_scores_over_time),
    ("/network-statistics", network_statistics),
]

for path, endpoint in routes:
    router.add_api_route(
        path,
        endpoint,
        tags=["retrieval"],
        dependencies=[Depends(verify_request_public)],
        methods=["GET"]
    )
