@echo off
echo Setting up environment...

REM Check if uv is initialized
if not exist uv.toml (
    pip install uv
    uv init
)

REM Check if virtual environment exists
if not exist .venv (
    uv venv .venv
)

uv sync
echo Finished setup.

if not exist tictactoe_model.pt (
    echo Building model...
    if not exist tictactoe_data.csv (
        if not exist data_generator.exe (
            echo Compiling data generator...
            g++ data_generator.cpp -o data_generator.exe
        )
        echo Generating data...
        data_generator.exe
    )

    echo Training model...
    uv run train.py
)

echo Running game...
uv run play.py
pause
