# Hammer Time : Evaluating Hierarchical Learning on Whack-a-Mole

This repository assesses the effectiveness of hierarchical learning, a learning technique which breaks down a complex task into smaller easily trainable parts, against a naive end-to-end approach.

Our environment is a simple gymnasium environment with a Fetch Mobile Manipulator and a hammer object, as well as several possible goal positions sampled from a grid. We shape our reward identically for both the hierarchical and end-to-end approaches.

## Requirements:

- Python 3.10.x
- Packages listed in `requirements.txt`

## Usage:

The following commands are provided:

- `train` : trains either end-to-end or partial task, saves checkpoint to specified directory
- `evaluate`: evaluates a model checkpoint(s) for either hierarchical or end-to-end actor on several seeded environment
- `see-envs` : saves videos of the different environments with a random actor

For more info, you can run `python -m whack_a_mole [COMMAND] -h`.

