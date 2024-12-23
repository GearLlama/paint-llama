from projen.python import PythonProject

AUTHOR = "Autonomous Llama"
AUTHOR_EMAIL = "paint-llama@gmail.com"
TASK_FLAG = "[PAINT_LLAMA]"
AWS_PROFILE = "paint-llama"

# Root project
root_project = PythonProject(
    author_name=AUTHOR,
    author_email=AUTHOR_EMAIL,
    module_name="",
    name="paint_llama",
    version="0.0.0",
    poetry=True,
    pytest=False,
    deps=[
        "python@3.12.7",
        "projen@0.91.1",
        "iac@{path = 'iac', develop = true}",
        "renderer@{path = 'renderer', develop = true}",
    ],
    dev_deps=["pre-commit", "flake8", "flake8-docstrings", "Flake8-pyproject", "pylint", "mypy", "black", "isort"],
)
root_project.add_git_ignore("**/cdk.out")
root_project.add_git_ignore("**/.DS_Store")
root_project.add_git_ignore("**/.env")
root_project.add_git_ignore("iac/*mapping.json")
root_project.add_git_ignore("cdk.context.json")
root_project.add_git_ignore("data/*")
root_project.add_git_ignore("**/.git_diff_cache")
root_project.add_git_ignore(".logs/")
root_project.add_git_ignore(".pkls/")
root_project.add_task(
    "lint",
    exec="pre-commit run --all-files",
    description=f"{TASK_FLAG} Lint all files with pre-commit",
)
root_project.add_task(
    "synth",
    exec="projen --no-post",
    description=f"{TASK_FLAG} Synthesize the project",
)

# IAC project
iac_project = PythonProject(
    author_name=AUTHOR,
    author_email=AUTHOR_EMAIL,
    module_name="iac",
    name="iac",
    version="0.0.0",
    parent=root_project,
    outdir="iac",
    poetry=True,
    deps=[
        "python@3.12.7",
        "aws-cdk-lib@^2.167.1",
        "pydantic@^2.9.2",
        "pydantic-settings@^2.6.1",
    ],
)
root_project.add_task(
    "cdk-deploy",
    exec=(
        f"export AWS_PROFILE={AWS_PROFILE} && cd iac && npx cdk deploy "
        f"--app 'python app.py' --require-approval never --asset-parallelism "
        f"--asset-prebuild false --concurrency 10"
    ),
    description=f"{TASK_FLAG} Deploy all CDK stacks",
)

# Renderer
renderer_project = PythonProject(
    author_name=AUTHOR,
    author_email=AUTHOR_EMAIL,
    module_name="renderer",
    name="renderer",
    version="0.0.0",
    parent=root_project,
    outdir="renderer",
    poetry=True,
    deps=[
        "python@3.12.7",
        "pydantic@^2.9.2",
        "pydantic-settings@^2.6.1",
        "numpy@^2.2.0",
        "opencv-python@^4.10.0.84",
        "scipy@^1.14.1",
        "Pillow@^11.0.0",
        "torch@2.5.1",
        "tensorboardX@^2.6.2.2",
        "tensorboard@^2.18.0",
    ],
)
root_project.add_task(
    "renderer:monitor",
    exec="tensorboard --logdir=.logs/renderer",
    description=f"{TASK_FLAG} Monitor the training process",
)
root_project.add_task(
    "renderer:train",
    exec="python renderer/renderer/train.py",
    description=f"{TASK_FLAG} Train the model",
)
root_project.add_task(
    "renderer:test",
    exec="python -m pytest renderer/tests",
    description=f"{TASK_FLAG} Run the tests",
)

root_project.synth()
iac_project.synth()
renderer_project.synth()
