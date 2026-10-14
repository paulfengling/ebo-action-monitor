"""Classification stability and optional declarative SOP, independent of the model."""


class StableAction:
    def __init__(self, threshold, confirmations):
        self.threshold, self.confirmations = threshold, confirmations
        self.reset()

    def reset(self):
        self.candidate, self.count = None, 0

    def update(self, label, confidence):
        if confidence < self.threshold:
            self.reset()
            return None
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1
        return label if self.count >= self.confirmations else None


class SOP:
    def __init__(self, cfg):
        self.enabled = cfg["sop"]["enabled"]
        self.steps = cfg["sop"]["steps"]
        self.kinds = {c["id"]: c["kind"] for c in cfg["classes"]}
        self.max_gap = cfg["stream"]["inference_interval"] + cfg["stream"]["max_result_age_seconds"]
        self.reset()

    def reset(self):
        self.index = 0
        self.durations = [0.0 for _ in self.steps]
        self.previous, self.previous_time = None, None
        self.failed = False

    def update(self, action, timestamp):
        alerts = []
        if action and self.kinds[action] == "incorrect":
            alerts.append("检测到已确认的错误动作")
            self.failed = True
        if self.enabled and self.index < len(self.steps) and action:
            step = self.steps[self.index]
            if action == step["action"]:
                if self.previous == action and self.previous_time is not None:
                    dt = timestamp - self.previous_time
                    if 0 <= dt <= self.max_gap:
                        self.durations[self.index] += dt
                if self.durations[self.index] >= step["min_seconds"]:
                    self.index += 1
            elif self.kinds[action] != "neutral" and not (
                self.index > 0 and action == self.steps[self.index-1]["action"]):
                alerts.append("步骤顺序不符或前一步持续时间不足")
                self.failed = True
        self.previous, self.previous_time = action, timestamp
        return {"enabled": self.enabled, "step_index": self.index, "estimated_seconds": list(self.durations),
                "status": "failed" if self.failed else "completed" if self.enabled and self.index == len(self.steps)
                else "in_progress" if self.enabled else "disabled", "alerts": alerts}
