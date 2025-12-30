# Those dictionaries are used for confusion matrix plots and result analysis
# They map action codes to human-readable action names

NTU_RGBD_60_ACTIONS = {
    "A030": "Type on keyboard",
    "A013": "Tear up paper",
    "A028": "Phone call",
    "A029": "Play with phone/tablet",
    "A032": "Taking a selfie",
    "A001": "Drink water",
    "A002": "Eat meal",
    "A004": "Brush hair",
    "A014": "Put on a jacket",
    "A015": "Take off a jacket",
    "A011": "Reading",
    "A012": "Writing",
    "A020": "Put on hat",
    "A021": "Take off hat",
    "A006": "Pick up"
}

NTU_RGBD_ACTIONS_ORDER = [
    "A001", "A002", "A004", "A006", "A011",
    "A012", "A013", "A014", "A015", "A020",
    "A021", "A028", "A029", "A030", "A032",
]

UNSAFE_NET_ORDERED = [
    "Safe Walkway Violation",
    "Unauthorized Intervention",
    "Opened Panel Cover",
    "Carrying Overload Forklift",
    "Safe Walkway",
    "Authorized Intervention",
    "Closed Panel Cover",
    "Safe Carrying"
]

NW_UCLA_ACTIONS = [
    "pick up with one hand",
    "pick up with two hands",
    "drop trash",
    "walk around",
    "sit down",
    "stand up",
    "donning",
    "doffing",
    "throw",
    "carry"
]
