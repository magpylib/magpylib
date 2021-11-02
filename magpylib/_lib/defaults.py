"""Package level config defaults"""

DEFAULTS = {
    "checkinputs": True,
    "edgesize": 1e-8,
    "itercylinder": 50,
    "display": {
        "autosizefactor": 10,
        "animation": {"maxfps": 30, "maxframes": 200},
        "backend": "matplotlib",
        "colorsequence": (
            "#2E91E5",
            "#E15F99",
            "#1CA71C",
            "#FB0D0D",
            "#DA16FF",
            "#222A2A",
            "#B68100",
            "#750D86",
            "#EB663B",
            "#511CFB",
            "#00A08B",
            "#FB00D1",
            "#FC0080",
            "#B2828D",
            "#6C7C32",
            "#778AAE",
            "#862A16",
            "#A777F1",
            "#620042",
            "#1616A7",
            "#DA60CA",
            "#6C4516",
            "#0D2A63",
            "#AF0038",
        ),
        "opacity": 1,
        "styles": {
            "base": {
                "path": {
                    "line": {"width": 1, "style": "solid", "color": None},
                    "marker": {"size": 1, "symbol": "o", "color": None},
                },
                "description": {"show": True, "text": None},
                "opacity": 1,
                "mesh3d": {"show": True, "data": None},
                "color": None,
            },
            "magnets": {
                "magnetization": {
                    "show": True,
                    "size": 1,
                    "color": {
                        "north": "#E71111",
                        "middle": "#DDDDDD",
                        "south": "#00B050",
                        "transition": 0.2,
                    },
                }
            },
            "currents": {"current": {"show": True, "size": 1}},
            "sensors": {"size": 1, "pixel": {"size": 1, "color": None, "symbol": "o"}},
            "dipoles": {"size": 1, "pivot": "middle"},
            "markers": {"marker": {"size": 2, "color": "grey", "symbol": "x"}},
        },
    },
}
