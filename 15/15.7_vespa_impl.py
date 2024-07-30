import os
import subprocess


def configure_vespa_cli(tenant_name: str):
    """Configure Vespa CLI for the given tenant name."""
    subprocess.run([
        "vespa", "config", "set", "target", "cloud"
    ], check=True)

    subprocess.run([
        "vespa", "config", "set", "--local", "application", f"{tenant_name}.vector-search.default"
    ], check=True)


def login_and_deploy():
    """Login to Vespa and deploy the configuration."""
    subprocess.run([
        "vespa", "auth", "login"
    ], check=True)

    subprocess.run([
        "vespa", "deploy"
    ], check=True)


def run_queries():
    """Run sample Vespa queries."""
    subprocess.run([
        "vespa", "query", "yql=select * from vectors where true",
        "ranking=unranked", "hits=1"
    ], check=True)

    subprocess.run([
        "vespa", "query", "yql=select * from vectors where true",
        "ranking=unranked", "hits=1",
        "summary=all", "presentation.format.tensors=short-value"
    ], check=True)


def visit_and_export_data():
    """Visit and export Vespa vector data to a JSONL file."""
    with open("../vector-data.jsonl", "w") as file:
        subprocess.run([
            "vespa", "visit", "--field-set", "vector:vector,id"
        ], stdout=file, check=True)


def main():
    """Main function to run the Vespa CLI commands."""
    tenant_name = "<tenant-name>"  # Replace with your tenant name

    configure_vespa_cli(tenant_name)
    login_and_deploy()
    run_queries()
    visit_and_export_data()


if __name__ == "__main__":
    main()
