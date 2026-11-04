"""Agrios's shared service layer.

The farm logic -- fields, tasks, weather, crop calendars -- lives here, once.
The chat agent's tools (main.py) and the web console both call these
functions, so a task logged from a button and a task logged from chat run the
same code and can't disagree. Services return typed models and raise typed
errors; turning those into what the model or a web page sees is the caller's
job, not theirs.
"""
