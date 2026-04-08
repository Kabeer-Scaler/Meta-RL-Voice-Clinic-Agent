"""
Server entry point for VoiceClinicAgent.

This module provides the FastAPI application for the OpenEnv environment.
It can be run with: uv run server
"""

import uvicorn
from app import app

def main():
    """Start the uvicorn server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
