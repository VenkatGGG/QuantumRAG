---
name: devops-worker
description: Worker for Docker containerization and infrastructure setup
---

# DevOps Worker

Handles Dockerfile creation, docker-compose configuration, and deployment setup.

## When to Use This Skill

Use for features involving:
- Dockerfile creation with multi-stage builds
- docker-compose.yml configuration
- Volume mounts for persistence
- Container optimization

## Required Skills

None

## Work Procedure

1. **Write tests first (if applicable)**
   - Create docker-compose test script
   - Test build process
   - Test volume persistence

2. **Create Dockerfile**
   - Use python:3.11-slim base (PyTorch compatibility)
   - Install system dependencies
   - Install Python requirements
   - Handle PyTorch binaries efficiently
   - Expose port 8000
   - Set correct CMD

3. **Create docker-compose.yml**
   - Build from Dockerfile
   - Expose port 8000:8000
   - Mount volume for HDF5 persistence at /app/data
   - Set environment variables if needed

4. **Test the setup**
   - Build image: `docker-compose build`
   - Start container: `docker-compose up -d`
   - Test endpoints: `curl http://localhost:8000/status`
   - Test persistence: restart container, verify data still exists
   - Stop container: `docker-compose down`

5. **Commit work**

## Example Handoff

```json
{
  "salientSummary": "Created Dockerfile with efficient PyTorch handling and docker-compose.yml with port 8000 exposed and volume mount at /app/data for HDF5 persistence across container restarts.",
  "whatWasImplemented": "Created Dockerfile using python:3.11-slim base with system dependencies (build-essential, libhdf5-dev). Installed PyTorch CPU version for smaller image size. Created docker-compose.yml building from Dockerfile, mapping host port 8000 to container port 8000, mounting ./data to /app/data for vector store persistence. Added .dockerignore to exclude unnecessary files from build context.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "docker-compose build", "exitCode": 0, "observation": "Image built successfully in 3m 42s, size 2.1GB"},
      {"command": "docker-compose up -d", "exitCode": 0, "observation": "Container started, healthcheck passed"},
      {"command": "curl -s http://localhost:8000/status | jq", "exitCode": 0, "observation": "{\"chunk_count\": 127, \"vector_dimensions\": 384}"},
      {"command": "docker-compose down && docker-compose up -d", "exitCode": 0, "observation": "Container restarted, HDF5 data persisted (chunk_count still 127)"},
      {"command": "docker-compose down", "exitCode": 0, "observation": "Container stopped cleanly"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "tests/test_docker.py", "cases": [
        {"name": "test_docker_build", "verifies": "Docker image builds successfully"},
        {"name": "test_container_starts", "verifies": "Container starts and healthcheck passes"},
        {"name": "test_port_exposed", "verifies": "Port 8000 accessible from host"},
        {"name": "test_volume_persistence", "verifies": "HDF5 data persists across restarts"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Docker build fails due to PyTorch installation issues
- Container fails to start
- Volume mount not working (data not persisting)
- Port conflicts on 8000
