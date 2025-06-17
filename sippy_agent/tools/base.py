"""
Base classes and interfaces for Sippy Agent tools.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import httpx

logger = logging.getLogger(__name__)


class SippyToolInput(BaseModel):
    """Base input schema for Sippy tools."""
    pass


class SippyBaseTool(BaseTool, ABC):
    """Base class for all Sippy Agent tools."""
    
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    args_schema: Type[BaseModel] = SippyToolInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """Execute the tool with the given arguments."""
        pass
    
    async def _arun(self, **kwargs: Any) -> str:
        """Async version of _run. Default implementation calls _run."""
        return self._run(**kwargs)


class ExampleTool(SippyBaseTool):
    """Example tool to demonstrate the structure."""
    
    name: str = "example_tool"
    description: str = "An example tool that echoes back the input"
    
    class ExampleInput(SippyToolInput):
        message: str = Field(description="Message to echo back")
    
    args_schema: Type[BaseModel] = ExampleInput
    
    def _run(self, message: str) -> str:
        """Echo back the input message."""
        return f"Echo: {message}"


# Placeholder for future Sippy API tools
class SippyJobAnalysisTool(SippyBaseTool):
    """Tool for analyzing CI jobs (placeholder for future implementation)."""
    
    name: str = "analyze_job"
    description: str = "Analyze a CI job for failures and issues"
    
    class JobAnalysisInput(SippyToolInput):
        job_id: str = Field(description="ID of the CI job to analyze")
        include_logs: bool = Field(default=False, description="Whether to include log analysis")
    
    args_schema: Type[BaseModel] = JobAnalysisInput
    
    def _run(self, job_id: str, include_logs: bool = False) -> str:
        """Analyze a CI job (placeholder implementation)."""
        return f"Job analysis for {job_id} would be implemented here. Include logs: {include_logs}"


class SippyTestFailureTool(SippyBaseTool):
    """Tool for analyzing test failures (placeholder for future implementation)."""

    name: str = "analyze_test_failures"
    description: str = "Analyze test failures for patterns and root causes"

    class TestFailureInput(SippyToolInput):
        test_name: str = Field(description="Name of the failing test")
        time_range: Optional[str] = Field(default=None, description="Time range for analysis (e.g., '7d', '30d')")

    args_schema: Type[BaseModel] = TestFailureInput

    def _run(self, test_name: str, time_range: Optional[str] = None) -> str:
        """Analyze test failures (placeholder implementation)."""
        return f"Test failure analysis for '{test_name}' over {time_range or 'default'} period would be implemented here."


class SippyProwJobSummaryTool(SippyBaseTool):
    """Tool for getting prow job run summaries from Sippy API."""

    name: str = "get_prow_job_summary"
    description: str = "Get a summary of a prow job run. Input: just the numeric job ID (e.g., 1934795512955801600)"

    # Add sippy_api_url as a proper field
    sippy_api_url: Optional[str] = Field(default=None, description="Sippy API base URL")

    class ProwJobSummaryInput(SippyToolInput):
        prow_job_run_id: str = Field(description="Numeric prow job run ID only (e.g., 1934795512955801600)")
        sippy_api_url: Optional[str] = Field(default=None, description="Sippy API base URL (optional, uses config if not provided)")

    args_schema: Type[BaseModel] = ProwJobSummaryInput

    def _run(self, prow_job_run_id: str, sippy_api_url: Optional[str] = None) -> str:
        """Get prow job run summary from Sippy API."""
        # Use provided URL or fall back to instance URL
        api_url = sippy_api_url or self.sippy_api_url

        if not api_url:
            return "Error: No Sippy API URL configured. Please set SIPPY_API_URL environment variable or provide sippy_api_url parameter."

        # Clean and validate the job ID - extract just the numeric part
        clean_job_id = str(prow_job_run_id).strip()
        # Extract just the numeric part if there's extra text
        import re
        job_id_match = re.search(r'\b(\d{10,})\b', clean_job_id)
        if job_id_match:
            clean_job_id = job_id_match.group(1)
        elif not clean_job_id.isdigit():
            return f"Error: Invalid job ID format. Expected numeric ID, got: {prow_job_run_id}"

        # Construct the API endpoint
        endpoint = f"{api_url.rstrip('/')}/api/job/run/summary"

        try:
            # Make the API request
            params = {"prow_job_run_id": clean_job_id}
            logger.info(f"Making request to {endpoint} with params: {params}")

            with httpx.Client(timeout=30.0) as client:
                response = client.get(endpoint, params=params)
                response.raise_for_status()

                data = response.json()

                # Format the response for better readability
                return self._format_job_summary(data)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting job summary: {e}")
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            logger.error(f"Request error getting job summary: {e}")
            return f"Error: Failed to connect to Sippy API at {api_url} - {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: Invalid JSON response from Sippy API"
        except Exception as e:
            logger.error(f"Unexpected error getting job summary: {e}")
            return f"Error: Unexpected error - {str(e)}"

    def _format_job_summary(self, data: Dict[str, Any]) -> str:
        """Format the job summary data for display."""
        if not data:
            return "No data returned from Sippy API"

        # Extract main fields
        job_id = data.get("id", "Unknown")
        job_name = data.get("name", "Unknown")
        release = data.get("release", "Unknown")
        cluster = data.get("cluster", "Unknown")
        start_time = data.get("startTime", "")
        duration_seconds = data.get("durationSeconds", 0)
        overall_result = data.get("overallResult", "Unknown")
        reason = data.get("reason", "Unknown")
        succeeded = data.get("succeeded", False)
        failed = data.get("failed", False)
        infrastructure_failure = data.get("infrastructureFailure", False)
        known_failure = data.get("knownFailure", False)
        test_count = data.get("testCount", 0)
        test_failure_count = data.get("testFailureCount", 0)
        variants = data.get("variants", [])
        url = data.get("url", "")
        testgrid_url = data.get("testGridURL", "")

        # Legacy fields for backward compatibility
        test_failures = data.get("testFailures", {})
        degraded_operators = data.get("degradedOperators", {})

        # Build formatted response
        result = f"**Prow Job Summary**\n\n"
        result += f"**Job ID:** {job_id}\n"
        result += f"**Job Name:** {job_name}\n"
        result += f"**Release:** {release}\n"
        result += f"**Cluster:** {cluster}\n\n"

        # Format timing information
        result += f"**⏱️ Timing & Duration:**\n"
        if start_time:
            # Parse and format the start time
            formatted_start = self._format_timestamp(start_time)
            result += f"Start Time: {formatted_start}\n"

        if duration_seconds > 0:
            formatted_duration = self._format_duration(duration_seconds)
            result += f"Duration: {formatted_duration} ({duration_seconds:,} seconds)\n"
        else:
            result += f"Duration: Not available\n"
        result += "\n"

        # Format results
        result += f"**📊 Results:**\n"
        result += f"Overall Result: {overall_result}\n"
        result += f"Succeeded: {'✅ Yes' if succeeded else '❌ No'}\n"
        result += f"Failed: {'❌ Yes' if failed else '✅ No'}\n"
        result += f"Infrastructure Failure: {'🚨 Yes' if infrastructure_failure else '✅ No'}\n"
        result += f"Known Failure: {'⚠️ Yes' if known_failure else '✅ No'}\n"
        result += f"Reason: {reason}\n\n"

        # Format test information
        result += f"**🧪 Test Information:**\n"
        result += f"Total Tests: {test_count}\n"
        result += f"Failed Tests: {test_failure_count}\n"
        if test_failure_count > 0 and test_count > 0:
            failure_rate = (test_failure_count / test_count) * 100
            result += f"Failure Rate: {failure_rate:.1f}%\n"
        result += "\n"

        # Format variants
        if variants:
            result += f"**🔧 Configuration Variants:**\n"
            # Group variants by type
            variant_groups = {}
            for variant in variants:
                if ':' in variant:
                    key, value = variant.split(':', 1)
                    variant_groups[key] = value
                else:
                    variant_groups['Other'] = variant_groups.get('Other', []) + [variant]

            for key, value in variant_groups.items():
                if isinstance(value, list):
                    result += f"{key}: {', '.join(value)}\n"
                else:
                    result += f"{key}: {value}\n"
            result += "\n"

        # Format legacy test failures if present
        if test_failures:
            result += f"**❌ Failed Tests Details ({len(test_failures)} total):**\n"
            for i, (test_name, failure_msg) in enumerate(test_failures.items(), 1):
                # Truncate very long failure messages
                truncated_msg = failure_msg[:200] + "..." if len(failure_msg) > 200 else failure_msg
                result += f"{i}. **{test_name}**\n"
                result += f"   Error: {truncated_msg}\n\n"

        # Format degraded operators if present
        if degraded_operators:
            result += f"**⚠️ Degraded Operators ({len(degraded_operators)} total):**\n"
            for i, (operator_name, operator_info) in enumerate(degraded_operators.items(), 1):
                result += f"{i}. **{operator_name}**\n"
                if isinstance(operator_info, str):
                    result += f"   Info: {operator_info}\n"
                else:
                    result += f"   Info: {str(operator_info)}\n"
                result += "\n"

        # Add useful links
        result += f"**🔗 Links:**\n"
        if url:
            result += f"[View Job in Prow]({url})\n"
        if testgrid_url:
            result += f"[View in TestGrid]({testgrid_url})\n"

        return result

    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp to a more readable format."""
        try:
            from datetime import datetime
            # Handle timezone offset format like "2025-06-16T22:09:31-04:00"
            if timestamp.endswith('Z'):
                dt = datetime.fromisoformat(timestamp[:-1])
            elif '+' in timestamp[-6:] or '-' in timestamp[-6:]:
                # Remove timezone for simple parsing
                dt = datetime.fromisoformat(timestamp[:-6])
            else:
                dt = datetime.fromisoformat(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            return timestamp

    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to a human-readable format."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            remaining_seconds = seconds % 60
            if remaining_seconds > 0:
                return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
            else:
                return f"{hours}h {remaining_minutes}m"


class SippyLogAnalyzerTool(SippyBaseTool):
    """Tool for analyzing job artifacts and logs from Sippy API using the /api/jobs/artifacts endpoint."""

    name: str = "analyze_job_logs"
    description: str = "Search job artifacts for patterns. Input: numeric job ID, optional path_glob and text_regex"

    # Add sippy_api_url as a proper field
    sippy_api_url: Optional[str] = Field(default=None, description="Sippy API base URL")

    class LogAnalyzerInput(SippyToolInput):
        prow_job_run_id: str = Field(description="Numeric prow job run ID only (e.g., 1934795512955801600)")
        path_glob: str = Field(
            default="*build-log*",
            description="Path glob pattern to match artifacts (e.g., '*build-log*', '*.log', '**/junit*.xml')"
        )
        text_regex: str = Field(
            default="[Ee]rror|[Ff]ail",
            description="Regex pattern to search for in the artifacts (e.g., '[Ee]rror', 'timeout', 'panic')"
        )
        sippy_api_url: Optional[str] = Field(default=None, description="Sippy API base URL (optional, uses config if not provided)")

    args_schema: Type[BaseModel] = LogAnalyzerInput

    def _run(self, prow_job_run_id: str, path_glob: str = "*build-log*",
             text_regex: str = "[Ee]rror|[Ff]ail",
             sippy_api_url: Optional[str] = None) -> str:
        """Fetch and analyze job artifacts from Sippy API."""
        # Use provided URL or fall back to instance URL
        api_url = sippy_api_url or self.sippy_api_url

        if not api_url:
            return "Error: No Sippy API URL configured. Please set SIPPY_API_URL environment variable or provide sippy_api_url parameter."

        # Clean and validate the job ID - ensure it's just the numeric ID
        clean_job_id = str(prow_job_run_id).strip()
        # Extract just the numeric part if there's extra text
        import re
        job_id_match = re.search(r'\b(\d{10,})\b', clean_job_id)
        if job_id_match:
            clean_job_id = job_id_match.group(1)
        elif not clean_job_id.isdigit():
            return f"Error: Invalid job ID format. Expected numeric ID, got: {prow_job_run_id}"

        # Construct the API endpoint
        endpoint = f"{api_url.rstrip('/')}/api/jobs/artifacts"

        try:
            # Make the API request with correct parameter names
            params = {
                "prowJobRuns": clean_job_id,  # Just the numeric ID
                "pathGlob": path_glob,
                "textRegex": text_regex
            }

            logger.info(f"Making request to {endpoint} with params: {params}")

            with httpx.Client(timeout=60.0) as client:  # Longer timeout for log analysis
                response = client.get(endpoint, params=params)
                response.raise_for_status()

                # The response should be JSON containing the matched artifacts
                data = response.json()

                # Format the response for better readability
                return self._format_log_analysis(data, clean_job_id, path_glob, text_regex)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error analyzing logs: {e}")
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            logger.error(f"Request error analyzing logs: {e}")
            return f"Error: Failed to connect to Sippy API at {api_url} - {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: Invalid JSON response from Sippy API"
        except Exception as e:
            logger.error(f"Unexpected error analyzing logs: {e}")
            return f"Error: Unexpected error - {str(e)}"

    def _format_log_analysis(self, data: Any, job_id: str, path_glob: str, regex: str) -> str:
        """Format the log analysis results for display."""
        if not data:
            return f"No artifacts found matching pattern '{path_glob}' with regex '{regex}' for job {job_id}"

        result = f"**Log Analysis Results**\n\n"
        result += f"**Job Run ID:** {job_id}\n"
        result += f"**Path Pattern:** {path_glob}\n"
        result += f"**Search Pattern:** {regex}\n\n"

        # Handle the new Sippy API response format
        if isinstance(data, dict) and "job_runs" in data:
            job_runs = data.get("job_runs", [])
            if not job_runs:
                result += "**Results:** No job runs found\n"
                return result

            for job_run in job_runs:
                artifacts = job_run.get("artifacts", [])
                if not artifacts:
                    result += "**Results:** No artifacts found\n"
                    continue

                result += f"**Found {len(artifacts)} matching artifacts:**\n\n"

                # Analyze each artifact
                for artifact in artifacts:
                    artifact_path = artifact.get("artifact_path", "unknown")
                    artifact_url = artifact.get("artifact_url", "")
                    matched_content = artifact.get("matched_content", {})

                    result += f"**📁 {artifact_path}**\n"
                    if artifact_url:
                        result += f"🔗 [View full log]({artifact_url})\n\n"

                    # Process line matches
                    line_matches = matched_content.get("line_matches", {})
                    matches = line_matches.get("matches", [])

                    if matches:
                        result += f"**Found {len(matches)} error/failure patterns:**\n\n"

                        # Analyze and categorize the errors
                        analysis = self._analyze_error_patterns(matches)
                        result += analysis

                        # Show first few raw matches for reference
                        result += "\n**Raw Error Lines:**\n"
                        for i, match_obj in enumerate(matches[:5], 1):
                            match_text = match_obj.get("match", str(match_obj))
                            # Clean up the match text
                            clean_match = match_text.strip().replace('\n', ' ')
                            if len(clean_match) > 200:
                                clean_match = clean_match[:200] + "..."
                            result += f"{i}. {clean_match}\n"

                        if len(matches) > 5:
                            result += f"... and {len(matches) - 5} more error lines\n"

                        if line_matches.get("truncated"):
                            result += "\n⚠️ *Results were truncated - there may be more errors*\n"
                    else:
                        result += "No error patterns found in this artifact\n"

                    result += "\n"
        else:
            # Fallback for other response formats
            result += f"**Results:**\n{str(data)[:500]}...\n"

        return result

    def _analyze_error_patterns(self, matches: list) -> str:
        """Analyze error patterns and provide insights."""
        analysis = "**🔍 Error Analysis:**\n"

        # Categorize errors
        registry_errors = []
        timeout_errors = []
        step_failures = []
        entrypoint_errors = []
        other_errors = []

        for match_obj in matches:
            match_text = match_obj.get("match", str(match_obj)).lower()

            if "registry" in match_text and ("503" in match_text or "unavailable" in match_text):
                registry_errors.append(match_obj)
            elif "timeout" in match_text or "timed out" in match_text:
                timeout_errors.append(match_obj)
            elif "step" in match_text and "failed" in match_text:
                step_failures.append(match_obj)
            elif "entrypoint" in match_text:
                entrypoint_errors.append(match_obj)
            else:
                other_errors.append(match_obj)

        # Provide analysis based on error types
        if registry_errors:
            analysis += f"\n🚨 **Registry Issues ({len(registry_errors)} occurrences):**\n"
            # Extract registry details
            for error in registry_errors[:2]:  # Show first 2
                error_text = error.get("match", "")
                if "registry.build11.ci.openshift.org" in error_text:
                    analysis += "- Problems with registry.build11.ci.openshift.org (503 Service Unavailable)\n"
                elif "registry" in error_text:
                    analysis += f"- Registry connectivity issue: {error_text[:100]}...\n"
            analysis += "💡 *This suggests infrastructure issues with the container registry*\n"

        if step_failures:
            analysis += f"\n⚠️ **Step Failures ({len(step_failures)} occurrences):**\n"
            failed_steps = set()
            for error in step_failures:
                error_text = error.get("match", "")
                if "step " in error_text.lower():
                    # Extract step name
                    import re
                    step_match = re.search(r'step ([a-zA-Z0-9-_]+)', error_text.lower())
                    if step_match:
                        failed_steps.add(step_match.group(1))

            for step in list(failed_steps)[:3]:  # Show first 3 unique steps
                analysis += f"- {step}\n"
            analysis += "💡 *Multiple pipeline steps failed, likely due to the registry issue*\n"

        if entrypoint_errors:
            analysis += f"\n🔄 **Process Issues ({len(entrypoint_errors)} occurrences):**\n"
            analysis += "- Test process execution failures\n"
            analysis += "- Process termination/interruption\n"
            analysis += "💡 *These are likely secondary failures caused by the primary issue*\n"

        if timeout_errors:
            analysis += f"\n⏱️ **Timeout Issues ({len(timeout_errors)} occurrences):**\n"
            analysis += "- Operations timed out\n"

        # Provide overall assessment
        analysis += f"\n**🎯 Root Cause Assessment:**\n"
        if registry_errors:
            analysis += "Primary issue appears to be registry connectivity problems (build11.ci.openshift.org returning 503 errors). "
            analysis += "This caused downstream failures in installation and test execution steps.\n"
        elif timeout_errors and step_failures:
            analysis += "Multiple timeouts and step failures suggest infrastructure or resource issues.\n"
        elif step_failures:
            analysis += "Multiple step failures suggest a systematic issue in the pipeline.\n"
        else:
            analysis += "Mixed error patterns - requires deeper investigation.\n"

        return analysis


class SippyJiraIncidentTool(SippyBaseTool):
    """Tool for querying Jira for known open incidents in the TRT project."""

    name: str = "check_known_incidents"
    description: str = "Check Jira for known open TRT incidents. Leave search_terms empty to see all incidents, or use specific keywords like 'registry' or 'timeout'."

    # Add Jira configuration as proper fields
    jira_url: str = Field(default="https://issues.redhat.com", description="Jira instance URL")
    jira_username: Optional[str] = Field(default=None, description="Jira username")
    jira_token: Optional[str] = Field(default=None, description="Jira API token")

    class JiraIncidentInput(SippyToolInput):
        search_terms: Optional[str] = Field(
            default=None,
            description="Optional search terms to filter incidents (e.g., 'registry', 'build11', 'timeout')"
        )
        jira_url: Optional[str] = Field(default=None, description="Jira URL (optional, uses config if not provided)")

    args_schema: Type[BaseModel] = JiraIncidentInput

    def _run(self, search_terms: Optional[str] = None, jira_url: Optional[str] = None) -> str:
        """Query Jira for known open incidents."""
        # Use provided URL or fall back to instance URL
        api_url = jira_url or self.jira_url

        if not api_url:
            return "Error: No Jira URL configured. Please set JIRA_URL environment variable or provide jira_url parameter."

        # Clean up search terms - filter out common LLM artifacts
        clean_search_terms = None
        if search_terms:
            # Remove common LLM response artifacts and clean up the input
            search_terms = str(search_terms).strip()

            # Skip if it contains common LLM artifacts or is too long
            skip_patterns = [
                "none", "null", "no job", "let's", "we can", "this time",
                "search for all", "open incidents", "trt project", "default value"
            ]

            if (len(search_terms) > 50 or
                any(pattern in search_terms.lower() for pattern in skip_patterns) or
                search_terms.lower() in ["none", "null", ""]):
                clean_search_terms = None
            else:
                # Extract meaningful keywords
                import re
                # Look for technical terms that might be relevant
                tech_terms = re.findall(r'\b(registry|build\d+|timeout|error|fail|503|502|infrastructure|node|cluster|network)\b', search_terms.lower())
                if tech_terms:
                    clean_search_terms = ','.join(tech_terms[:3])  # Limit to 3 terms
                else:
                    clean_search_terms = None

        # Construct the Jira REST API endpoint
        endpoint = f"{api_url.rstrip('/')}/rest/api/2/search"

        # Build JQL query for TRT project incidents
        jql_parts = [
            'project = "TRT"',
            'labels = "trt-incident"',
            'status not in (Closed, Done, Resolved)'
        ]

        # Add search terms if we have clean ones
        if clean_search_terms:
            # Split search terms and add them as text search
            terms = [term.strip() for term in clean_search_terms.split(',') if term.strip()]
            if terms:
                text_search = ' OR '.join([f'text ~ "{term}"' for term in terms])
                jql_parts.append(f'({text_search})')

        jql = ' AND '.join(jql_parts)

        try:
            # Prepare request parameters
            params = {
                'jql': jql,
                'fields': 'key,summary,status,priority,created,updated,description,labels',
                'maxResults': 20  # Limit results
            }

            # Prepare authentication if available
            auth = None
            if self.jira_username and self.jira_token:
                auth = (self.jira_username, self.jira_token)

            if clean_search_terms:
                logger.info(f"Querying Jira with search terms: {clean_search_terms}")
            else:
                logger.info("Querying Jira for all open TRT incidents")
            logger.info(f"JQL: {jql}")

            # Make the API request
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    endpoint,
                    params=params,
                    auth=auth,
                    headers={'Accept': 'application/json'}
                )
                response.raise_for_status()

                data = response.json()

                # Format the response
                return self._format_jira_incidents(data, clean_search_terms)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error querying Jira: {e}")
            if e.response.status_code == 401:
                return "Error: Jira authentication failed. Check JIRA_USERNAME and JIRA_TOKEN environment variables."
            elif e.response.status_code == 403:
                return "Error: Access denied to Jira. You may need authentication or permissions to view TRT project."
            else:
                return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            logger.error(f"Request error querying Jira: {e}")
            return f"Error: Failed to connect to Jira at {api_url} - {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: Invalid JSON response from Jira API"
        except Exception as e:
            logger.error(f"Unexpected error querying Jira: {e}")
            return f"Error: Unexpected error - {str(e)}"

    def _format_jira_incidents(self, data: Dict[str, Any], search_terms: Optional[str] = None) -> str:
        """Format the Jira incidents for display."""
        issues = data.get('issues', [])
        total = data.get('total', 0)

        if not issues:
            if search_terms:
                return f"No open TRT incidents found matching search terms: {search_terms}"
            else:
                return "No open TRT incidents found with 'trt-incident' label"

        result = f"**Known Open Incidents**\n\n"
        if search_terms:
            result += f"**Search Terms:** {search_terms}\n"
        result += f"**Found {len(issues)} of {total} total incidents:**\n\n"

        for issue in issues:
            key = issue.get('key', 'Unknown')
            fields = issue.get('fields', {})
            summary = fields.get('summary', 'No summary')
            status = fields.get('status', {}).get('name', 'Unknown')
            priority = fields.get('priority', {}).get('name', 'Unknown')
            created = fields.get('created', '')
            updated = fields.get('updated', '')
            description = fields.get('description', '')
            labels = fields.get('labels', [])

            # Format dates
            created_date = self._format_jira_date(created)
            updated_date = self._format_jira_date(updated)

            result += f"**🎫 {key}** - {summary}\n"
            result += f"📊 Status: {status} | Priority: {priority}\n"
            result += f"📅 Created: {created_date} | Updated: {updated_date}\n"

            if labels:
                relevant_labels = [label for label in labels if 'trt' in label.lower() or 'incident' in label.lower()]
                if relevant_labels:
                    result += f"🏷️ Labels: {', '.join(relevant_labels)}\n"

            # Include description snippet if available
            if description:
                # Clean and truncate description
                clean_desc = description.replace('\n', ' ').replace('\r', '').strip()
                if len(clean_desc) > 200:
                    clean_desc = clean_desc[:200] + "..."
                result += f"📝 Description: {clean_desc}\n"

            # Add Jira link
            jira_base = self.jira_url.rstrip('/')
            result += f"🔗 [View in Jira]({jira_base}/browse/{key})\n\n"

        if total > len(issues):
            result += f"... and {total - len(issues)} more incidents (use more specific search terms to narrow results)\n"

        return result

    def _format_jira_date(self, date_str: str) -> str:
        """Format Jira date string to a more readable format."""
        if not date_str:
            return "Unknown"

        try:
            # Jira dates are in ISO format like "2024-01-15T10:30:45.000+0000"
            from datetime import datetime
            # Remove timezone info for simple parsing
            clean_date = date_str.split('T')[0]
            date_obj = datetime.strptime(clean_date, '%Y-%m-%d')
            return date_obj.strftime('%Y-%m-%d')
        except Exception:
            return date_str.split('T')[0] if 'T' in date_str else date_str
