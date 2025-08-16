# Troubleshooting

This page lists common problems and their solutions.

## Bot not appearing in the game

*   **Check that Rocket League is running.**
*   **Check that the RLBot GUI is running.**
*   **Check that the bot is enabled in the RLBot GUI.**
*   **Check the bot's logs for errors.** The logs are located in the `logs` directory.

## Bot is not moving or is behaving unexpectedly

*   **Check that the correct model is loaded.** The `model_path` in `src/bot.cfg` should point to a valid model file.
*   **Check that the observation builder and action parser are configured correctly.** The `obs_builder` and `act_parser` in `src/bot.cfg` should match the ones used to train the model.
*   **Check the bot's logs for errors.** The logs are located in the `logs` directory.

## "Could not import class..." error

This error means that the bot could not find the specified class.

*   **Check that the class name is spelled correctly** in `src/bot.cfg`.
*   **Check that the file containing the class is in the correct directory.** For example, the `DefaultObs` class should be in the `src/obs/default_obs.py` file.
*   **Check that the file name matches the class name.** For example, the `DefaultObs` class should be in a file named `default_obs.py`.

## "ModuleNotFoundError: No module named 'rlgym'"

This error means that the `rlgym` library is not installed.

*   **Make sure you have installed the required packages.** Run `pip install -r requirements.txt` to install all the required packages.
