"""Heuristic control policy for train control.

Simple rule-based agent for baseline performance and demonstration.
"""


def heuristic(env, s):
    """Heuristic policy for train control.
    
    A rule-based policy that:
    1. Notches up when significantly below speed limit
    2. Notches down when over speed limit
    3. Makes fine adjustments based on acceleration
    4. Anticipates upcoming speed limit changes
    
    Parameters
    ----------
    env : gym.Env
        The environment (unused but kept for compatibility)
    s : array-like
        State vector with components:
        s[0] = train speed (mph)
        s[1] = train acceleration (mph/minute)
        s[2] = current location (miles)
        s[3] = current speed limit (mph)
        s[4] = next speed limit (mph)
        s[5] = next speed limit location (miles)
        
    Returns
    -------
    int
        Action: 0=hold, 1=notch up, 2=notch down
        
    Examples
    --------
    >>> action = heuristic(env, state)
    """
    train_speed = s[0]
    train_acceleration = s[1]
    current_loc = s[2]
    current_speed_limit = s[3]
    next_speed_limit = s[4]
    next_speed_limit_loc = s[5]

    # Drive to the current speed limit unless the next speed limit is lower and within 3 miles
    speed_limit = current_speed_limit
    if next_speed_limit < current_speed_limit and next_speed_limit_loc < current_loc + 3:
        speed_limit = next_speed_limit

    # Rule 1: Far below speed limit - notch up aggressively
    if train_speed < speed_limit - 10:
        return 1

    # Rule 2: Over speed limit - notch down
    if train_speed > speed_limit:
        return 2

    # Rule 3: Approaching speed limit with low acceleration - notch up
    if speed_limit - 5 < train_speed < speed_limit - 2 and train_acceleration <= 1:
        return 1

    # Rule 4: Near speed limit and accelerating - notch down preemptively
    if speed_limit - 3 < train_speed < speed_limit - 1 and train_acceleration > 0:
        return 2

    # Default: hold current notch
    return 0
