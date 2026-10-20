"""All recognisers, each in its own folder with everything about it (code, labels, data, models, runs).

Every recogniser folder has:
  __init__.py   FOLDER, NAME and DESCRIPTION (shown in main.py's menu)
  app.py        main(): live recognition       (main.py ... run)
  collect.py    main(): record training data   (main.py ... collect)
  train.py      main(): train on that data     (main.py ... train)

To add a new recogniser: make a folder like that (copy recognisers/gestures as a starting point),
then add it to RECOGNISERS below. main.py's menu then shows it automatically.
"""
import importlib

# menu key -> package
RECOGNISERS = {
    "gestures": "recognisers.gestures",
    "items": "recognisers.items",
}

# action -> the module in each recogniser folder whose main() does it
ACTION_MODULES = {
    "run": "app",
    "collect": "collect",
    "train": "train",
}


def load(key):
    """Import one recogniser's __init__.py (cheap: just its name and description)."""
    return importlib.import_module(RECOGNISERS[key])


def action_main(key, action):
    """The main() function for one action. Its module (and heavy libraries like PyTorch) only loads now."""
    return importlib.import_module(f"{RECOGNISERS[key]}.{ACTION_MODULES[action]}").main
