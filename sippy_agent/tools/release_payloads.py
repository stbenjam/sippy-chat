"""
Tool for getting OpenShift release payload information from the release controller API.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type
from pydantic import Field
import httpx

from .base_tool import SippyBaseTool, SippyToolInput

logger = logging.getLogger(__name__)


class SippyReleasePayloadTool(SippyBaseTool):
    """Tool for getting OpenShift release payload information."""
    
    name: str = "get_release_payloads"
    description: str = "Get OpenShift release payload information and basic status. When asked for 'latest' or 'last' payload, returns the most recent payload's name and status. For basic payload info, use this tool first. Input: release version (e.g., '4.20') and optional stream type ('nightly' or 'ci', defaults to 'nightly')"
    
    # Release controller API base URL
    release_controller_url: str = Field(
        default="https://amd64.ocp.releases.ci.openshift.org/api/v1",
        description="Release controller API base URL"
    )
    
    class ReleasePayloadInput(SippyToolInput):
        release_version: str = Field(description="Release version (e.g., '4.20', '4.19')")
        stream_type: Optional[str] = Field(
            default="nightly", 
            description="Stream type: 'nightly' or 'ci' (defaults to 'nightly')"
        )
        include_ready: Optional[bool] = Field(
            default=False,
            description="Include 'Ready' phase payloads (defaults to False)"
        )
        limit: Optional[int] = Field(
            default=10,
            description="Maximum number of payloads to return (defaults to 10)"
        )
        payload_name: Optional[str] = Field(
            default=None,
            description="Specific payload name to get basic status for (e.g., '4.20.0-0.nightly-2025-06-17-061341')"
        )
    
    args_schema: Type[SippyToolInput] = ReleasePayloadInput
    
    def _run(
        self,
        release_version: str,
        stream_type: Optional[str] = "nightly",
        include_ready: Optional[bool] = False,
        limit: Optional[int] = 10,
        payload_name: Optional[str] = None
    ) -> str:
        """Get release payload information from the release controller API."""

        # If a specific payload name is provided, get basic status for that payload
        if payload_name:
            # Clean the payload name in case it includes parameter syntax
            clean_payload_name = self._clean_payload_name(payload_name)
            return self._get_payload_basic_status(clean_payload_name)

        # Validate and clean inputs
        stream_type = stream_type or "nightly"
        if stream_type not in ["nightly", "ci"]:
            return f"Error: Invalid stream type '{stream_type}'. Must be 'nightly' or 'ci'."
        
        # Clean release version (remove any extra characters)
        clean_version = re.sub(r'[^\d\.]', '', release_version)
        if not re.match(r'^\d+\.\d+$', clean_version):
            return f"Error: Invalid release version format. Expected format like '4.20', got: {release_version}"
        
        # Construct the release stream name
        release_stream = f"{clean_version}.0-0.{stream_type}"
        
        # Construct the API endpoint
        endpoint = f"{self.release_controller_url.rstrip('/')}/releasestream/{release_stream}/tags"
        
        try:
            logger.info(f"Making request to {endpoint}")
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(endpoint)
                response.raise_for_status()
                
                data = response.json()
                
                # Format the response
                return self._format_payload_response(
                    data, 
                    release_version=clean_version,
                    stream_type=stream_type,
                    include_ready=include_ready,
                    limit=limit
                )
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting release payloads: {e}")
            if e.response.status_code == 404:
                return f"Error: Release stream '{release_stream}' not found. Check if the release version and stream type are correct."
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            logger.error(f"Request error getting release payloads: {e}")
            return f"Error: Failed to connect to release controller API - {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: Invalid JSON response from release controller API"
        except Exception as e:
            logger.error(f"Unexpected error getting release payloads: {e}")
            return f"Error: Unexpected error - {str(e)}"

    def _format_payload_response(
        self, 
        data: Dict[str, Any], 
        release_version: str,
        stream_type: str,
        include_ready: bool,
        limit: int
    ) -> str:
        """Format the payload response data for display."""
        if not data:
            return "No data returned from release controller API"

        release_stream_name = data.get("name", "Unknown")
        tags = data.get("tags", [])
        
        if not tags:
            return f"No payloads found for release stream {release_stream_name}"

        # Filter payloads based on include_ready flag
        filtered_tags = []
        for tag in tags:
            phase = tag.get("phase", "").lower()
            if include_ready or phase != "ready":
                filtered_tags.append(tag)
        
        # Limit the number of results
        if limit and limit > 0:
            filtered_tags = filtered_tags[:limit]

        # Build formatted response
        result = f"**OpenShift Release Payloads - {release_version} {stream_type.title()}**\n\n"
        result += f"**Release Stream:** {release_stream_name}\n"
        result += f"**Total Payloads:** {len(tags)} (showing {len(filtered_tags)})\n\n"

        if not filtered_tags:
            result += "No payloads found matching the criteria.\n"
            if not include_ready:
                result += "Note: 'Ready' phase payloads are excluded by default. Use include_ready=True to see them.\n"
            return result

        # Find the most recent payload for quick answer
        most_recent = filtered_tags[0] if filtered_tags else None
        if most_recent:
            phase = most_recent.get("phase", "Unknown")
            name = most_recent.get("name", "Unknown")
            result += f"**🎯 Latest {release_version} {stream_type} Payload:** {name}\n"
            result += f"**Status:** {phase}\n\n"

            # Add a clear answer format for direct questions
            result += f"**Quick Answer:** The last {release_version} {stream_type} payload was {name} and it was {phase.lower()}.\n\n"

        # List all payloads
        result += f"**📋 Payload List:**\n"
        for i, tag in enumerate(filtered_tags, 1):
            name = tag.get("name", "Unknown")
            phase = tag.get("phase", "Unknown")
            pull_spec = tag.get("pullSpec", "")
            download_url = tag.get("downloadURL", "")
            
            # Extract timestamp from name if possible
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{6})', name)
            timestamp_str = ""
            if timestamp_match:
                timestamp_raw = timestamp_match.group(1)
                # Format as YYYY-MM-DD HH:MM:SS
                if len(timestamp_raw) == 15:  # YYYY-MM-DD-HHMMSS
                    formatted_time = f"{timestamp_raw[:4]}-{timestamp_raw[5:7]}-{timestamp_raw[8:10]} {timestamp_raw[11:13]}:{timestamp_raw[13:15]}:00"
                    timestamp_str = f" ({formatted_time})"

            # Status emoji
            status_emoji = {
                "accepted": "✅",
                "rejected": "❌", 
                "ready": "🔄",
                "failed": "💥"
            }.get(phase.lower(), "❓")

            result += f"{i}. **{name}**{timestamp_str}\n"
            result += f"   Status: {status_emoji} {phase}\n"
            
            if pull_spec:
                result += f"   Pull Spec: `{pull_spec}`\n"
            if download_url:
                result += f"   [Download]({download_url})\n"
            result += "\n"

        # Add summary statistics
        phase_counts = {}
        for tag in filtered_tags:
            phase = tag.get("phase", "Unknown").lower()
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        if phase_counts:
            result += f"**📊 Status Summary:**\n"
            for phase, count in sorted(phase_counts.items()):
                emoji = {
                    "accepted": "✅",
                    "rejected": "❌", 
                    "ready": "🔄",
                    "failed": "💥"
                }.get(phase, "❓")
                result += f"{emoji} {phase.title()}: {count}\n"

        return result

    def get_latest_payload(self, release_version: str, stream_type: str = "nightly") -> Optional[Dict[str, Any]]:
        """Helper method to get just the latest payload information."""
        try:
            # Use the main _run method but parse the result differently
            # This is a simplified version for programmatic access
            clean_version = re.sub(r'[^\d\.]', '', release_version)
            release_stream = f"{clean_version}.0-0.{stream_type}"
            endpoint = f"{self.release_controller_url.rstrip('/')}/releasestream/{release_stream}/tags"
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(endpoint)
                response.raise_for_status()
                data = response.json()
                
                tags = data.get("tags", [])
                if not tags:
                    return None
                
                # Find first non-Ready payload
                for tag in tags:
                    if tag.get("phase", "").lower() != "ready":
                        return tag
                
                # If all are Ready, return the first one
                return tags[0] if tags else None
                
        except Exception as e:
            logger.error(f"Error getting latest payload: {e}")
            return None

    def _get_payload_basic_status(self, payload_name: str) -> str:
        """Get basic status information for a specific payload."""
        try:
            # Extract release stream from payload name
            release_stream = self._extract_release_stream(payload_name)
            if not release_stream:
                return f"Error: Could not extract release stream from payload name '{payload_name}'. Expected format like '4.20.0-0.nightly-2025-06-17-061341'"

            # Get the payload list to find this specific payload
            endpoint = f"{self.release_controller_url.rstrip('/')}/releasestream/{release_stream}/tags"

            with httpx.Client(timeout=30.0) as client:
                response = client.get(endpoint)
                response.raise_for_status()
                data = response.json()

                tags = data.get("tags", [])

                # Find the specific payload
                target_payload = None
                for tag in tags:
                    if tag.get("name") == payload_name:
                        target_payload = tag
                        break

                if not target_payload:
                    return f"Error: Payload '{payload_name}' not found in release stream '{release_stream}'"

                # Format basic status
                phase = target_payload.get("phase", "Unknown")
                pull_spec = target_payload.get("pullSpec", "")

                # Extract timestamp from name if possible
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{6})', payload_name)
                timestamp_str = ""
                if timestamp_match:
                    timestamp_raw = timestamp_match.group(1)
                    if len(timestamp_raw) == 15:  # YYYY-MM-DD-HHMMSS
                        formatted_time = f"{timestamp_raw[:4]}-{timestamp_raw[5:7]}-{timestamp_raw[8:10]} {timestamp_raw[11:13]}:{timestamp_raw[13:15]}:00"
                        timestamp_str = f" (Created: {formatted_time})"

                # Status emoji
                status_emoji = {
                    "accepted": "✅",
                    "rejected": "❌",
                    "ready": "🔄",
                    "failed": "💥"
                }.get(phase.lower(), "❓")

                result = f"**OpenShift Release Payload Status**\n\n"
                result += f"**Payload:** {payload_name}{timestamp_str}\n"
                result += f"**Status:** {status_emoji} {phase}\n"
                result += f"**Release Stream:** {release_stream}\n"

                if pull_spec:
                    result += f"**Pull Spec:** `{pull_spec}`\n"

                # Add contextual message based on status
                if phase.lower() == "rejected":
                    result += f"\n❌ **This payload was rejected due to blocking job failures.**\n"
                    result += f"💡 **Want to know why?** Use `get_payload_details` with payload name `{payload_name}` for detailed failure analysis.\n"
                elif phase.lower() == "accepted":
                    result += f"\n✅ **This payload was accepted and is ready for use.**\n"
                elif phase.lower() == "ready":
                    result += f"\n🔄 **This payload is ready but not yet tested by blocking jobs.**\n"

                return result

        except Exception as e:
            logger.error(f"Error getting payload basic status: {e}")
            return f"Error: Failed to get payload status - {str(e)}"

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
