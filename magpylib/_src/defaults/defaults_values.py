"""Package level config defaults"""

DEFAULTS = {
    "display": {
        "autosizefactor": 10,
        "animation": {
            "fps": 20,
            "maxfps": 30,
            "maxframes": 200,
            "time": 5,
            "slider": True,
        },
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
        "style": {
            "base": {
                "path": {
                    "line": {"width": 1, "style": "solid", "color": None},
                    "marker": {"size": 2, "symbol": "o", "color": None},
                    "show": True,
                    "frames": None,
                    "numbering": False,
                },
                "description": {"show": True, "text": None},
                "opacity": 1,
                "model3d": {"showdefault": True, "data": []},
                "color": None,
            },
            "magnet": {
                "magnetization": {
                    "show": True,
                    "size": 1,
                    "color": {
                        "north": "#E71111",
                        "middle": "#DDDDDD",
                        "south": "#00B050",
                        "transition": 0.2,
                        "mode": "tricolor",
                    },
                }
            },
            "current": {"arrow": {"show": True, "size": 1, "width": 2}},
            "sensor": {
                "size": 1,
                "pixel": {"size": 1, "color": None, "symbol": "o"},
                "arrows": {
                    "x": {"color": "red"},
                    "y": {"color": "green"},
                    "z": {"color": "blue"},
                },
            },
            "dipole": {"size": 1, "pivot": "middle"},
            "markers": {"marker": {"size": 2, "color": "grey", "symbol": "x"}},
        },
    },
}
