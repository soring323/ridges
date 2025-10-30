# Test Analysis Tool - Helps agent understand tests without running them

def analyze_test_spec(test_spec_string):
    """
    Parse a test specification string and extract useful information.
    
    Example:
    "astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]"
    
    Returns:
    {
        'file': 'astropy/modeling/tests/test_separable.py',
        'function': 'test_separable',
        'parameters': 'compound_model6-result6',
        'likely_fixture': 'compound_model6'
    }
    """
    import re
    
    result = {}
    
    # Extract file path
    if '::' in test_spec_string:
        parts = test_spec_string.split('::')
        result['file'] = parts[0]
        
        # Extract function and parameters
        if len(parts) > 1:
            func_part = parts[1]
            
            # Check for parametrized test
            param_match = re.match(r'(\w+)\[(.*)\]', func_part)
            if param_match:
                result['function'] = param_match.group(1)
                result['parameters'] = param_match.group(2)
                
                # Try to extract fixture name
                param_parts = result['parameters'].split('-')
                if param_parts:
                    result['likely_fixture'] = param_parts[0]
            else:
                result['function'] = func_part
    
    return result


def extract_test_guidance(fail_to_pass, pass_to_pass):
    """
    Generate guidance for the agent based on test specifications.
    
    Returns a string that explains what the agent should focus on.
    """
    guidance = []
    
    guidance.append("# Test-Driven Debugging Guide\n")
    
    if fail_to_pass:
        guidance.append(f"\n## Tests That Must Pass ({len(fail_to_pass)} tests)")
        guidance.append("These tests are currently FAILING and must PASS after your fix:\n")
        
        seen_files = set()
        for test in fail_to_pass:
            info = analyze_test_spec(test)
            file = info.get('file', 'unknown')
            
            if file not in seen_files:
                guidance.append(f"\n### File: {file}")
                guidance.append(f"   → Read this file to understand expected behavior")
                seen_files.add(file)
            
            guidance.append(f"   - {info.get('function', 'test')} with {info.get('parameters', 'no params')}")
            
            if info.get('likely_fixture'):
                guidance.append(f"     (check fixture/test data: '{info['likely_fixture']}')")
    
    if pass_to_pass:
        guidance.append(f"\n## Tests That Must Stay Passing ({len(pass_to_pass)} tests)")
        guidance.append("These tests are currently PASSING - DO NOT BREAK THEM:\n")
        
        # Just show count and files to avoid clutter
        files = set()
        for test in pass_to_pass:
            info = analyze_test_spec(test)
            if info.get('file'):
                files.add(info['file'])
        
        for file in sorted(files):
            guidance.append(f"   - {file}")
    
    guidance.append("\n## Recommended Approach:")
    guidance.append("1. Read the failing test code first (fail_to_pass)")
    guidance.append("2. Understand what behavior it expects")
    guidance.append("3. Find the code being tested (usually imported in test file)")
    guidance.append("4. Make MINIMAL changes to fix the specific issue")
    guidance.append("5. Mentally verify pass_to_pass tests won't break")
    
    return '\n'.join(guidance)


# Add this as a tool in sam.py:
def create_test_guidance_tool(test_specs):
    """
    Factory function to create a test guidance tool with test specs.
    """
    
    @EnhancedToolManager.tool
    def get_test_guidance(self):
        '''
        Get guidance about which tests need to pass and recommended debugging approach.
        This tool analyzes the test specifications without running them.
        Call this FIRST to understand what you need to fix.
        '''
        fail_to_pass = test_specs.get('fail_to_pass', [])
        pass_to_pass = test_specs.get('pass_to_pass', [])
        
        return extract_test_guidance(fail_to_pass, pass_to_pass)
    
    return get_test_guidance


# Example usage in tool manager initialization:
"""
if test_specs:
    # Add test guidance tool
    guidance_tool = create_test_guidance_tool(test_specs)
    tool_manager.register_tool(guidance_tool)
"""
