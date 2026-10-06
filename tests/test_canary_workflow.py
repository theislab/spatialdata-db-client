"""Test publish-catalog staleness canary workflow."""

from pathlib import Path

import yaml


def test_publish_catalog_workflow_exists():
    """Test that the publish-catalog workflow file exists."""
    workflow_file = Path(__file__).parent.parent / ".github" / "workflows" / "publish-catalog.yml"
    assert workflow_file.exists(), f"Workflow file {workflow_file} does not exist"


def test_publish_catalog_workflow_yaml_valid():
    """Test that the workflow file parses as valid YAML."""
    workflow_file = Path(__file__).parent.parent / ".github" / "workflows" / "publish-catalog.yml"
    with open(workflow_file) as f:
        workflow = yaml.safe_load(f)
    assert workflow is not None, "Workflow YAML does not parse"


def test_publish_catalog_workflow_has_triggers():
    """Test that the workflow has workflow_dispatch and schedule triggers."""
    workflow_file = Path(__file__).parent.parent / ".github" / "workflows" / "publish-catalog.yml"
    with open(workflow_file) as f:
        workflow = yaml.safe_load(f)

    # Note: yaml.safe_load() parses 'on' as the boolean True
    assert True in workflow, "Workflow is missing 'on' (parsed as True) key"
    on = workflow[True]

    # Check for workflow_dispatch trigger
    assert "workflow_dispatch" in on, "Workflow is missing 'workflow_dispatch' trigger"

    # Check for schedule trigger
    assert "schedule" in on, "Workflow is missing 'schedule' trigger"


def test_publish_catalog_workflow_has_canary_step():
    """Test that the workflow has a step downloading catalog.parquet and reading sddb_generated_at."""
    workflow_file = Path(__file__).parent.parent / ".github" / "workflows" / "publish-catalog.yml"
    with open(workflow_file) as f:
        workflow = yaml.safe_load(f)

    assert "jobs" in workflow, "Workflow is missing 'jobs' key"
    jobs = workflow["jobs"]
    assert len(jobs) > 0, "Workflow has no jobs"

    # Find a job with a step referencing catalog.parquet
    found_download = False
    found_sddb_generated_at = False

    for _job_name, job in jobs.items():
        if "steps" in job:
            for step in job["steps"]:
                step_str = str(step)
                if "catalog.parquet" in step_str:
                    found_download = True
                if "sddb_generated_at" in step_str:
                    found_sddb_generated_at = True

    assert found_download, "Workflow does not have a step referencing 'catalog.parquet'"
    assert found_sddb_generated_at, "Workflow does not have a step referencing 'sddb_generated_at'"
