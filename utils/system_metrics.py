import psutil
import utils.logger as logger

from typing import Optional
from pydantic import BaseModel
from utils.docker import get_num_docker_containers



class SystemMetrics(BaseModel):
    """
    cpu_percent: CPU percentage (0-100)
    ram_percent: RAM percentage (0-100)
    ram_total_gb: Total RAM in GB
    disk_percent: Disk percentage (0-100)
    disk_total_gb: Total disk in GB
    num_containers: Number of Docker containers
    """

    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    ram_total_gb: Optional[float] = None
    disk_percent: Optional[float] = None
    disk_total_gb: Optional[float] = None
    num_containers: Optional[int] = None



async def get_system_metrics() -> SystemMetrics:
    metrics = SystemMetrics()

    try:
        metrics.cpu_percent = psutil.cpu_percent()

        memory = psutil.virtual_memory()
        metrics.ram_percent = memory.percent
        metrics.ram_total_gb = memory.total / (1000 ** 3)

        disk = psutil.disk_usage('/')
        metrics.disk_percent = disk.percent
        metrics.disk_total_gb = disk.total / (1000 ** 3)

        metrics.num_containers = get_num_docker_containers()

    except Exception as e:
        logger.warning(f"Error in get_system_metrics(): {e}")
        
    return metrics