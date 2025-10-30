# Workaround for missing dependencies in agent sandbox
# This can be added to the agent's initialization

import sys
from types import ModuleType

def patch_missing_dependencies():
    """
    Create mock modules for common missing dependencies
    to prevent import errors during test discovery.
    """
    
    # Mock setuptools_scm (for astropy version detection)
    if 'setuptools_scm' not in sys.modules:
        mock_scm = ModuleType('setuptools_scm')
        mock_scm.get_version = lambda **kwargs: "0.0.0.dev0"
        sys.modules['setuptools_scm'] = mock_scm
    
    # Mock extension_helpers (another common astropy dependency)
    if 'extension_helpers' not in sys.modules:
        mock_helpers = ModuleType('extension_helpers')
        sys.modules['extension_helpers'] = mock_helpers
    
    # Suppress warnings about missing optional dependencies
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='astropy')
    warnings.filterwarnings('ignore', message='.*setuptools_scm.*')

# Call this in sam.py before running tests
if __name__ == "__main__":
    patch_missing_dependencies()
    print("✅ Patched missing dependencies")
