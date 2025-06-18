"""
Tool for getting detailed OpenShift release payload information from the release controller API.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type
from pydantic import Field
import httpx

from .base_tool import SippyBaseTool, SippyToolInput

logger = logging.getLogger(__name__)


class SippyPayloadDetailsTool(SippyBaseTool):
    """Tool for getting detailed OpenShift release payload information."""
    
    name: str = "get_payload_details"
    description: str = "Get detailed failure information for a specific OpenShift release payload including failed blocking jobs and their prow job IDs. Shows which jobs failed but does NOT automatically suggest log analysis. Use this ONLY when user asks for details about WHY a payload failed. For basic payload status, use get_release_payloads first. Input: payload name (e.g., '4.20.0-0.nightly-2025-06-17-061341')"
    
    # Release controller API base URL
    release_controller_url: str = Field(
        default="https://amd64.ocp.releases.ci.openshift.org/api/v1",
        description="Release controller API base URL"
    )

    # Sippy API URL for job analysis
    sippy_api_url: Optional[str] = Field(
        default=None,
        description="Sippy API base URL for job analysis"
    )
    
    class PayloadDetailsInput(SippyToolInput):
        payload_name: str = Field(description="Full payload name (e.g., '4.20.0-0.nightly-2025-06-17-061341')")
        include_job_analysis: Optional[bool] = Field(
            default=False,
            description="Include suggested next steps for analyzing failed blocking jobs"
        )
        max_jobs_to_analyze: Optional[int] = Field(
            default=5,
            description="Maximum number of failed jobs to analyze in detail (defaults to 5 to avoid excessive API calls)"
        )
    
    args_schema: Type[SippyToolInput] = PayloadDetailsInput
    
    def _run(
        self,
        payload_name: str,
        include_job_analysis: Optional[bool] = False,
        max_jobs_to_analyze: Optional[int] = 5
    ) -> str:
        """Get detailed payload information from the release controller API."""

        # Clean the payload name in case it includes parameter syntax
        clean_payload_name = self._clean_payload_name(payload_name)

        # Extract release stream from payload name
        release_stream = self._extract_release_stream(clean_payload_name)
        if not release_stream:
            return f"Error: Could not extract release stream from payload name '{clean_payload_name}'. Expected format like '4.20.0-0.nightly-2025-06-17-061341'"
        
        # Construct the API endpoint for payload details
        endpoint = f"{self.release_controller_url.rstrip('/')}/releasestream/{release_stream}/release/{clean_payload_name}"
        
        try:
            logger.info(f"Making request to {endpoint}")

            with httpx.Client(timeout=30.0) as client:
                response = client.get(endpoint)
                response.raise_for_status()

                # Log response details for debugging
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response content type: {response.headers.get('content-type', 'unknown')}")

                # Check if response is JSON (be more lenient with content type checking)
                content_type = response.headers.get('content-type', '')
                if not (content_type.startswith('application/json') or content_type.startswith('text/json') or response.text.strip().startswith('{')):
                    logger.warning(f"Unexpected content type: {content_type}")
                    logger.warning(f"Response text: {response.text[:500]}...")
                    return f"Error: API returned non-JSON response. Content-Type: {content_type}"

                try:
                    data = response.json()
                    logger.debug(f"JSON parsed successfully, type: {type(data)}")
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON decode error: {json_err}")
                    logger.error(f"Response text: {response.text[:500]}...")
                    return f"Error: Invalid JSON response from API. Response: {response.text[:200]}..."

                # Validate that data is a dictionary
                if not isinstance(data, dict):
                    logger.error(f"Expected dict, got {type(data)}: {str(data)[:200]}...")
                    return f"Error: API returned unexpected data type {type(data)}. Expected JSON object."

                # Format the response
                return self._format_payload_details(data, clean_payload_name, include_job_analysis, max_jobs_to_analyze)
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting payload details: {e}")
            if e.response.status_code == 404:
                return f"Error: Payload '{clean_payload_name}' not found in release stream '{release_stream}'. Check if the payload name is correct."
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            logger.error(f"Request error getting payload details: {e}")
            return f"Error: Failed to connect to release controller API - {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: Invalid JSON response from release controller API"
        except Exception as e:
            logger.error(f"Unexpected error getting payload details: {e}")
            return f"Error: Unexpected error - {str(e)}"

    def _clean_payload_name(self, payload_name: str) -> str:
        """Clean payload name from common parameter syntax issues."""
        # Remove common parameter syntax patterns
        cleaned = payload_name.strip()

        # Handle cases like "payload name = '4.20.0-0.nightly-2025-06-17-061341'"
        if '=' in cleaned:
            cleaned = cleaned.split('=')[-1].strip()

        # Remove quotes
        cleaned = cleaned.strip('\'"')

        # Extract just the payload name pattern
        payload_pattern = re.search(r'(\d+\.\d+\.0-0\.(nightly|ci)-\d{4}-\d{2}-\d{2}-\d{6})', cleaned)
        if payload_pattern:
            return payload_pattern.group(1)

        return cleaned

    def _extract_release_stream(self, payload_name: str) -> Optional[str]:
        """Extract release stream from payload name."""
        # Expected format: 4.20.0-0.nightly-2025-06-17-061341
        match = re.match(r'^(\d+\.\d+\.0-0\.(nightly|ci))-\d{4}-\d{2}-\d{2}-\d{6}$', payload_name)
        if match:
            return match.group(1)
        return None

    def _format_payload_details(self, data: Dict[str, Any], payload_name: str, include_job_analysis: bool, max_jobs_to_analyze: int = 5) -> str:
        """Format the detailed payload response data for display."""
        if not data:
            return "No data returned from release controller API"

        # Validate data structure
        if not isinstance(data, dict):
            logger.error(f"Expected dict in _format_payload_details, got {type(data)}")
            return f"Error: Invalid data format received from API"

        try:
            name = data.get("name", payload_name)
            phase = data.get("phase", "Unknown")
            results = data.get("results", {})
            upgrades_to = data.get("upgradesTo", [])
            change_log = data.get("changeLog", {})
        except Exception as e:
            logger.error(f"Error extracting data fields: {e}")
            logger.error(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            return f"Error: Failed to parse payload data structure - {str(e)}"

        try:
            # Build concise formatted response
            result = f"**Payload Analysis: {name}**\n\n"
            result += f"**Status:** {self._get_status_emoji(phase)} {phase}\n\n"

            # Analyze blocking jobs if payload was rejected/failed
            blocking_jobs = results.get("blockingJobs", {})
            if blocking_jobs:
                failed_jobs = []
                for job_name, job_info in blocking_jobs.items():
                    if not isinstance(job_info, dict):
                        continue
                    state = job_info.get("state", "Unknown")
                    if state.lower() == "failed":
                        url = job_info.get("url", "")
                        prow_job_id = self._extract_prow_job_id(url)
                        failed_jobs.append((job_name, prow_job_id, url))

                # Summary
                total_blocking = len(blocking_jobs)
                failed_count = len(failed_jobs)
                result += f"**Summary:** {failed_count} out of {total_blocking} blocking jobs failed\n\n"

                if failed_jobs:
                    result += f"**Failed Blocking Jobs:**\n"
                    for job_name, prow_job_id, url in failed_jobs:
                        result += f"• **{job_name}**\n"
                        if prow_job_id:
                            result += f"  Job ID: `{prow_job_id}`\n"
                        result += "\n"

                    # Include analysis suggestions for failed jobs only if requested
                    if include_job_analysis:
                        failed_jobs_dict = {job[0]: {"url": job[2]} for job in failed_jobs}
                        result += self._suggest_job_analysis(failed_jobs_dict, max_jobs_to_analyze)

            return result

        except Exception as e:
            logger.error(f"Error formatting payload details: {e}")
            logger.error(f"Data structure: {str(data)[:200]}...")
            return f"Error: Failed to format payload details - {str(e)}"

    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        status_emojis = {
            "accepted": "✅",
            "rejected": "❌",
            "failed": "💥",
            "ready": "🔄",
            "pending": "⏳",
            "running": "🏃"
        }
        return status_emojis.get(status.lower(), "❓")

    def _extract_prow_job_id(self, url: str) -> Optional[str]:
        """Extract prow job ID from URL."""
        if not url:
            return None
        
        # URL format: https://prow.ci.openshift.org/view/gs/test-platform-results/logs/.../1934869209162977280
        match = re.search(r'/(\d{10,})/?$', url)
        if match:
            return match.group(1)
        return None

    def _suggest_job_analysis(self, blocking_jobs: Dict[str, Any], max_jobs: int = 5) -> str:
        """Provide concise job analysis guidance."""
        prow_job_ids = []
        for job_name, job_info in blocking_jobs.items():
            url = job_info.get("url", "")
            prow_job_id = self._extract_prow_job_id(url)
            if prow_job_id:
                prow_job_ids.append((job_name, prow_job_id))

        if prow_job_ids:
            jobs_to_analyze = min(max_jobs, len(prow_job_ids))
            result = f"**Next Steps:** Analyze {jobs_to_analyze} key failed jobs:\n"

            for i, (job_name, job_id) in enumerate(prow_job_ids[:jobs_to_analyze], 1):
                result += f"{i}. Job ID `{job_id}` ({job_name.split('-')[-1]})\n"

            result += f"\nFor each job: get_prow_job_summary → analyze_job_logs → look for patterns\n"
            return result

        return ""
