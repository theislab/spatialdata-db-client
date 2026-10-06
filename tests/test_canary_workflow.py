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
    """Test that the workflow reads Catalog().generated_at and checks staleness."""
    workflow_file = Path(__file__).parent.parent / ".github" / "workflows" / "publish-catalog.yml"
    with open(workflow_file) as f:
        workflow = yaml.safe_load(f)

    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    text = "\n".join(str(step.get("run", "")) for step in steps)
    assert "Catalog().generated_at" in text, "Workflow does not read Catalog().generated_at"
    assert "stale" in text, "Workflow does not check staleness"
