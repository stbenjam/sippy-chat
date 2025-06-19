"""
Tool for parsing JUnit XML files and extracting test failures and flakes.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Type
from pydantic import Field
import httpx

from .base_tool import SippyBaseTool, SippyToolInput

logger = logging.getLogger(__name__)


class JUnitParserTool(SippyBaseTool):
    """Tool for parsing JUnit XML files to extract test failures and flakes."""
    
    name: str = "parse_junit_xml"
    description: str = "Parse JUnit XML file from URL to get test failures and flakes. Takes junit_xml_url as required parameter and optional test_name parameter."
    
    class JUnitParserInput(SippyToolInput):
        junit_xml_url: str = Field(description="URL to the JUnit XML file")
        test_name: Optional[str] = Field(default=None, description="Optional specific test name to get results for")
    
    args_schema: Type[SippyToolInput] = JUnitParserInput
    
    def _run(self, junit_xml_url: str, test_name: Optional[str] = None) -> str:
        """Parse JUnit XML file and extract test failures and flakes."""
        try:
            # Handle case where agent passes JSON string instead of parsed parameters
            if junit_xml_url.startswith('{') and junit_xml_url.endswith('}'):
                try:
                    import json
                    parsed = json.loads(junit_xml_url)
                    if 'junit_xml_url' in parsed:
                        junit_xml_url = parsed['junit_xml_url']
                        if 'test_name' in parsed:
                            test_name = parsed['test_name']
                except json.JSONDecodeError:
                    return f"Error: Received malformed JSON input: {junit_xml_url}"

            # Fetch the XML content
            logger.info(f"Fetching JUnit XML from: {junit_xml_url}")
            
            with httpx.Client(timeout=60.0) as client:
                response = client.get(junit_xml_url)
                response.raise_for_status()
                
                xml_content = response.text
                
            # Parse the XML
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                logger.error(f"XML parse error: {e}")
                return f"Error: Invalid XML format - {str(e)}"
            
            # Extract test results
            test_results = self._extract_test_results(root)
            
            # Process results based on requirements
            if test_name:
                # Return all results for the specific test (no overall size limit for specific tests)
                filtered_results = [result for result in test_results if result['name'] == test_name]
                if not filtered_results:
                    return f"No test results found for test name: {test_name}"
                return self._format_test_results(filtered_results, test_name)
            else:
                # Return only failures and flakes, limit to 25 but respect 150KB overall limit
                failures_and_flakes = self._identify_failures_and_flakes(test_results)

                # Format results while respecting the 150KB overall limit
                result_text, actual_count, total_count = self._format_test_results_with_limit(failures_and_flakes, max_size_kb=150)

                if actual_count < total_count:
                    result_text += f"\n\n**Note:** Results truncated to {actual_count} entries due to size limit. Total failures/flakes found: {total_count}"
                elif len(failures_and_flakes) > 25:
                    result_text += f"\n\n**Note:** Results truncated to first 25 entries. Total failures/flakes found: {len(failures_and_flakes)}"

                return result_text
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching JUnit XML: {e}")
            return f"Error: HTTP {e.response.status_code} - Failed to fetch XML from {junit_xml_url}"
        except httpx.RequestError as e:
            logger.error(f"Request error fetching JUnit XML: {e}")
            return f"Error: Failed to connect to {junit_xml_url} - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error parsing JUnit XML: {e}")
            return f"Error: Unexpected error - {str(e)}"
    
    def _extract_test_results(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract all test results from the JUnit XML."""
        test_results = []
        
        # Handle different JUnit XML structures
        # Look for testcase elements at various levels
        testcases = []
        
        # Direct testcase children
        testcases.extend(root.findall('.//testcase'))
        
        for testcase in testcases:
            test_name = testcase.get('name', 'Unknown')
            classname = testcase.get('classname', '')
            time_str = testcase.get('time', '0')
            
            # Parse duration
            try:
                duration = float(time_str)
            except (ValueError, TypeError):
                duration = 0.0
            
            # Determine test result
            failure = testcase.find('failure')
            error = testcase.find('error')
            skipped = testcase.find('skipped')
            
            if failure is not None:
                status = 'failure'
                output = failure.text or failure.get('message', '')
            elif error is not None:
                status = 'failure'
                output = error.text or error.get('message', '')
            elif skipped is not None:
                status = 'skipped'
                output = skipped.text or skipped.get('message', '')
            else:
                status = 'success'
                output = ''
            
            # Get system-out and system-err if available
            system_out = testcase.find('system-out')
            system_err = testcase.find('system-err')
            
            additional_output = []
            if system_out is not None and system_out.text:
                additional_output.append(f"STDOUT:\n{system_out.text}")
            if system_err is not None and system_err.text:
                additional_output.append(f"STDERR:\n{system_err.text}")
            
            if additional_output:
                if output:
                    output += "\n\n" + "\n\n".join(additional_output)
                else:
                    output = "\n\n".join(additional_output)
            
            # Truncate output to first 15KB
            if len(output) > 15360:  # 15KB
                output = output[:15360] + "\n... [output truncated to 15KB]"
            
            test_results.append({
                'name': test_name,
                'classname': classname,
                'full_name': f"{classname}.{test_name}" if classname else test_name,
                'duration': duration,
                'status': status,
                'output': output
            })
        
        return test_results
    
    def _identify_failures_and_flakes(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify failures and flakes from test results."""
        # Group tests by full name to identify flakes
        test_groups = {}
        for result in test_results:
            full_name = result['full_name']
            if full_name not in test_groups:
                test_groups[full_name] = []
            test_groups[full_name].append(result)
        
        failures_and_flakes = []
        
        for full_name, results in test_groups.items():
            if len(results) == 1:
                # Single test run
                result = results[0]
                if result['status'] in ['failure', 'error']:
                    failures_and_flakes.append(result)
            else:
                # Multiple test runs - check for flakes
                statuses = [r['status'] for r in results]
                success_count = statuses.count('success')
                failure_count = len([s for s in statuses if s in ['failure', 'error']])
                
                if success_count > 0 and failure_count > 0:
                    # This is a flake
                    # Create a combined result
                    total_duration = sum(r['duration'] for r in results)
                    combined_output = []
                    
                    for i, result in enumerate(results):
                        combined_output.append(f"Run {i+1} ({result['status']}):")
                        if result['output']:
                            combined_output.append(result['output'])
                        combined_output.append("")
                    
                    output_text = "\n".join(combined_output)
                    if len(output_text) > 15360:  # 15KB
                        output_text = output_text[:15360] + "\n... [output truncated to 15KB]"
                    
                    flake_result = {
                        'name': results[0]['name'],
                        'classname': results[0]['classname'],
                        'full_name': full_name,
                        'duration': total_duration,
                        'status': 'flake',
                        'success_count': success_count,
                        'failure_count': failure_count,
                        'output': output_text
                    }
                    failures_and_flakes.append(flake_result)
                elif failure_count > 0:
                    # All failures - add the first failure
                    failure_result = next(r for r in results if r['status'] in ['failure', 'error'])
                    failures_and_flakes.append(failure_result)
        
        return failures_and_flakes
    
    def _format_test_results(self, results: List[Dict[str, Any]], specific_test: Optional[str] = None) -> str:
        """Format test results for display."""
        if not results:
            if specific_test:
                return f"No results found for test: {specific_test}"
            else:
                return "No test failures or flakes found in the JUnit XML file."
        
        if specific_test:
            header = f"**Test Results for: {specific_test}**\n\n"
        else:
            header = f"**JUnit Test Failures and Flakes**\n\n"
            header += f"Found {len(results)} test failures/flakes:\n\n"
        
        formatted_results = []
        
        for i, result in enumerate(results, 1):
            test_info = f"**{i}. {result['name']}**\n"
            
            if result['classname']:
                test_info += f"   **Class:** {result['classname']}\n"
            
            test_info += f"   **Duration:** {result['duration']:.2f}s\n"
            
            if result['status'] == 'flake':
                test_info += f"   **Result:** FLAKE ({result['success_count']} successes, {result['failure_count']} failures)\n"
            else:
                test_info += f"   **Result:** {result['status'].upper()}\n"
            
            if result['output']:
                test_info += f"   **Output:**\n```\n{result['output']}\n```\n"
            
            formatted_results.append(test_info)
        
        return header + "\n".join(formatted_results)

    def _format_test_results_with_limit(self, results: List[Dict[str, Any]], max_size_kb: int = 150) -> tuple[str, int, int]:
        """Format test results while respecting overall size limit.

        Returns:
            tuple: (formatted_text, actual_count, total_count)
        """
        if not results:
            return "No test failures or flakes found in the JUnit XML file.", 0, 0

        max_size_bytes = max_size_kb * 1024
        total_count = len(results)

        # Limit to 25 results initially
        limited_results = results[:25]

        header = f"**JUnit Test Failures and Flakes**\n\n"
        header += f"Found {len(limited_results)} test failures/flakes:\n\n"

        formatted_results = []
        current_size = len(header.encode('utf-8'))
        actual_count = 0

        for i, result in enumerate(limited_results, 1):
            test_info = f"**{i}. {result['name']}**\n"

            if result['classname']:
                test_info += f"   **Class:** {result['classname']}\n"

            test_info += f"   **Duration:** {result['duration']:.2f}s\n"

            if result['status'] == 'flake':
                test_info += f"   **Result:** FLAKE ({result['success_count']} successes, {result['failure_count']} failures)\n"
            else:
                test_info += f"   **Result:** {result['status'].upper()}\n"

            if result['output']:
                test_info += f"   **Output:**\n```\n{result['output']}\n```\n"

            test_info += "\n"

            # Check if adding this test would exceed the size limit
            test_size = len(test_info.encode('utf-8'))
            if current_size + test_size > max_size_bytes and actual_count > 0:
                # Stop adding tests if we would exceed the limit
                break

            formatted_results.append(test_info)
            current_size += test_size
            actual_count += 1

        return header + "\n".join(formatted_results), actual_count, total_count
