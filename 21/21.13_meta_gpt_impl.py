# pip install --upgrade metagpt
# or `pip install --upgrade git+https://github.com/geekan/MetaGPT.git`
# or `git clone https://github.com/geekan/MetaGPT && cd MetaGPT && pip install --upgrade -e .`
# metagpt --init-config

# CLI:
# metagpt "Create a 2048 game"  # this will create a repo in ./workspace

# Use as library:
from metagpt.software_company import generate_repo, ProjectRepo

repo: ProjectRepo = generate_repo("Create a 2048 game")  # or ProjectRepo("<path>")
print(repo)  # it will print the repo structure with files

import asyncio
from metagpt.roles.di.data_interpreter import DataInterpreter


async def main():
    di = DataInterpreter()
    await di.run("Run data analysis on sklearn Iris dataset, include a plot")


asyncio.run(main())  # or await main() in a jupyter notebook setting
