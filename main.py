"""Master file: choose a recogniser, then run it live, collect training data, or train it.

Run with a menu:
  python main.py

Or go straight to an action:
  python main.py gestures run        live hand gesture recognition
  python main.py gestures collect    record gesture samples
  python main.py gestures train      train the gesture model
  python main.py items run           live item recognition (CLIP)
  python main.py items collect       record item photos
  python main.py items train         train the item model
  python main.py check               check libraries and webcam

Each recogniser lives in its own folder under recognisers/ (see recognisers/__init__.py).
"""
import argparse

from recognisers import RECOGNISERS, action_main, load

ACTIONS = {
    "run": "Run live recognition",
    "collect": "Collect training data",
    "train": "Train on collected data",
}


def ask(title, options):
    """Numbered menu in the terminal. options: [(key, text)]. Returns the chosen key, or None for back/quit."""
    print(f"\n{title}")
    for number, (_, text) in enumerate(options, start=1):
        print(f"  {number}. {text}")
    print("  0. Back / quit")
    while True:
        answer = input("Choose a number: ").strip().lstrip("﻿")   # ignore an invisible marker some terminals add
        if answer == "0":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        print("Please type one of the numbers above.")


def do(recogniser_key, action):
    """Run one action of one recogniser."""
    print(f"\n=== {load(recogniser_key).NAME}: {ACTIONS[action]} ===")
    action_main(recogniser_key, action)()


def check():
    from common.check_setup import main as check_setup
    check_setup()


def menu():
    """Keep offering choices until the user quits, so you can collect, train and run in one go."""
    while True:
        recogniser_options = [(key, f"{load(key).NAME}: {load(key).DESCRIPTION}") for key in RECOGNISERS]
        recogniser_key = ask("Which recogniser?", recogniser_options + [("check", "Check setup (libraries + webcam)")])
        if recogniser_key is None:
            print("Bye!")
            return
        if recogniser_key == "check":
            check()
            continue
        while True:
            action = ask(f"{load(recogniser_key).NAME}: what do you want to do?", list(ACTIONS.items()))
            if action is None:
                break
            # Stay in the menu if e.g. the webcam is busy. Direct commands still crash with the full error.
            try:
                do(recogniser_key, action)
            except (RuntimeError, FileNotFoundError) as error:
                print(f"\nProblem: {error}")


def main():
    parser = argparse.ArgumentParser(description="Choose a recogniser and run, collect or train it.")
    parser.add_argument("recogniser", nargs="?", choices=[*RECOGNISERS, "check"],
                        help="which recogniser (leave out for the menu)")
    parser.add_argument("action", nargs="?", choices=list(ACTIONS), help="what to do")
    args = parser.parse_args()

    if args.recogniser is None:
        menu()
    elif args.recogniser == "check":
        check()
    elif args.action is None:
        parser.error(f"choose an action for {args.recogniser}: {', '.join(ACTIONS)}")
    else:
        do(args.recogniser, args.action)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:   # Ctrl+C: quit quietly instead of printing an error
        print("\nStopped.")
