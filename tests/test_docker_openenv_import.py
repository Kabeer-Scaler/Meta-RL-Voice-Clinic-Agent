"""
Bug Condition Exploration Test for OpenEnv Module Import Failure

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

This test explores the bug condition where the Docker container fails to import
openenv-core modules on Hugging Face Spaces deployment.

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

The test encodes the expected behavior - it will validate the fix when it passes
after implementation.

OPTIMIZED: Reduced to 2 essential tests for faster execution.
"""

import subprocess
import sys


def test_docker_container_imports_openenv():
    """
    Combined test that verifies Docker build and openenv import in one test.
    
    **Property 1: Bug Condition** - OpenEnv Module Import Failure on Docker Build
    
    This test verifies:
    1. Docker build completes successfully
    2. openenv-core package is installed
    3. Imports work correctly: create_fastapi_app and Environment
    
    EXPECTED ON UNFIXED CODE: This test will FAIL with ModuleNotFoundError
    because openenv module uses underscore naming (openenv_core) not dot notation.
    """
    # Build the Docker image (only once)
    print("\n=== Building Docker Image ===")
    build_result = subprocess.run(
        ["docker", "build", "-t", "voiceclinic-test", "."],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    assert build_result.returncode == 0, (
        f"Docker build failed with return code {build_result.returncode}\n"
        f"STDERR: {build_result.stderr}"
    )
    
    # Verify package is installed
    print("\n=== Checking Package Installation ===")
    pip_result = subprocess.run(
        ["docker", "run", "--rm", "voiceclinic-test", "pip", "list"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert "openenv-core" in pip_result.stdout.lower(), (
        "openenv-core package not found in pip list"
    )
    print("✓ openenv-core package is installed")
    
    # Test import create_fastapi_app
    print("\n=== Testing Import: create_fastapi_app ===")
    import_result = subprocess.run(
        [
            "docker", "run", "--rm", "voiceclinic-test",
            "python", "-c", 
            "from openenv_core.env_server import create_fastapi_app; print('Import successful')"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if import_result.returncode != 0:
        print("\n=== COUNTEREXAMPLE FOUND ===")
        print(f"Import failed: {import_result.stderr}")
        if "No module named 'openenv'" in import_result.stderr:
            print("Confirmed: Module naming mismatch (openenv.core vs openenv_core)")
    
    assert import_result.returncode == 0, (
        f"Failed to import create_fastapi_app\n"
        f"STDERR: {import_result.stderr}\n"
        f"This confirms the bug: openenv module is not importable"
    )
    
    assert "Import successful" in import_result.stdout
    print("✓ create_fastapi_app import successful")
    
    # Test import Environment
    print("\n=== Testing Import: Environment ===")
    env_result = subprocess.run(
        [
            "docker", "run", "--rm", "voiceclinic-test",
            "python", "-c",
            "from openenv_core.env_server import Environment; print('Import successful')"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert env_result.returncode == 0, (
        f"Failed to import Environment\n"
        f"STDERR: {env_result.stderr}"
    )
    
    assert "Import successful" in env_result.stdout
    print("✓ Environment import successful")


def test_docker_container_starts_application():
    """
    Test that application starts without ModuleNotFoundError.
    
    **Property 1: Bug Condition** - OpenEnv Module Import Failure on Docker Build
    
    EXPECTED ON UNFIXED CODE: This test will FAIL because uvicorn cannot load
    the app module due to the import error.
    """
    # Ensure image is built
    subprocess.run(
        ["docker", "build", "-t", "voiceclinic-test", "."],
        capture_output=True,
        timeout=300
    )
    
    # Start container
    print("\n=== Starting Application Container ===")
    start_result = subprocess.run(
        [
            "docker", "run", "-d", "--name", "voiceclinic-test-container",
            "-p", "7860:7860", "voiceclinic-test"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    container_id = start_result.stdout.strip()
    print(f"Container ID: {container_id}")
    
    try:
        # Wait for startup
        import time
        time.sleep(3)
        
        # Check logs
        logs_result = subprocess.run(
            ["docker", "logs", "voiceclinic-test-container"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        logs_combined = logs_result.stdout + logs_result.stderr
        
        if "ModuleNotFoundError" in logs_combined:
            print("\n=== COUNTEREXAMPLE FOUND ===")
            print("Application crashed with ModuleNotFoundError")
            for line in logs_combined.split('\n'):
                if 'ModuleNotFoundError' in line or 'openenv' in line.lower():
                    print(line)
        
        # Check if running
        inspect_result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", "voiceclinic-test-container"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        is_running = inspect_result.stdout.strip() == "true"
        print(f"Container running: {is_running}")
        
        assert is_running, (
            f"Container not running. Application failed to start.\n"
            f"Logs:\n{logs_combined}"
        )
        
        assert "ModuleNotFoundError" not in logs_combined, (
            f"ModuleNotFoundError in logs:\n{logs_combined}"
        )
        
        print("✓ Application started successfully")
        
    finally:
        # Cleanup
        subprocess.run(
            ["docker", "stop", "voiceclinic-test-container"],
            capture_output=True,
            timeout=30
        )
        subprocess.run(
            ["docker", "rm", "voiceclinic-test-container"],
            capture_output=True,
            timeout=30
        )
