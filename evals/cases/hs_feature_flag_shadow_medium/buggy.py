class FeatureGate:
    def __init__(self) -> None:
        self._enabled: bool | None = None

    def is_enabled(self, config: dict[str, bool]) -> bool:
        if self._enabled is None:
            self._enabled = bool(config.get("flag", False))
        if "override" in config:
            enabled = bool(config["override"])
        return self._enabled
