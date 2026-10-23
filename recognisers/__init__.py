"""All recognisers, each in its own folder with everything about it (code, labels, data, models, runs).

Every recogniser folder has:
  __init__.py   FOLDER, NAME and DESCRIPTION (shown in main.py's menu)
  app.py        main(): live recognition       (main.py ... run)
  collect.py    main(): record training data   (main.py ... collect)   optional
  train.py      main(): train on that data     (main.py ... train)     optional

To add a new recogniser: make a folder like that (copy recognisers/gestures as a starting point),
then add it to RECOGNISERS below. main.py's menu then shows it, with only the actions it has.
"""
import importlib
import importlib.util

# menu key -> package
RECOGNISERS = {
    "gestures": "recognisers.gestures",
    "items": "recognisers.items",
    "faces": "recognisers.faces",
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


def actions(key):
    """The actions one recogniser has, e.g. the face tracer has only "run". Finds the files without loading them."""
    return [action for action, module in ACTION_MODULES.items()
            if importlib.util.find_spec(f"{RECOGNISERS[key]}.{module}") is not None]


def action_main(key, action):
    """The main() function for one action. Its module (and heavy libraries like PyTorch) only loads now."""
    return importlib.import_module(f"{RECOGNISERS[key]}.{ACTION_MODULES[action]}").main
