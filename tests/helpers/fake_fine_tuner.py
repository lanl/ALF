"""A stand-in fine-tuning adapter that records what it was asked to do, for tests without MACE."""
import json
from pathlib import Path


class FakeFineTuner:
    def __init__(self, settings):
        self.settings = settings

    def validate(self, master_config):
        if master_config.get("reject_fine_tuning"):
            raise ValueError("fake adapter rejects this run")

    def identity(self):
        return dict(package="fake-tuner", package_version="1.0")

    def prepare_data(self, shards, member_dir):
        path = Path(member_dir) / "data.json"
        path.write_text(json.dumps([Path(shard).name for shard in shards]))
        return str(path)

    def train_member(self, data, member_dir, member, seed, device, environment):
        import time

        if seed in self.settings.get("failing_seeds", []):
            raise RuntimeError(f"seed {seed} diverged")
        started = time.monotonic()
        time.sleep(self.settings.get("train_seconds", 0))
        (Path(member_dir) / "trained.json").write_text(json.dumps(dict(
            source=self.settings["foundation_model"], member=member, seed=seed, device=device,
            visible=environment.get("CUDA_VISIBLE_DEVICES"), data=json.loads(Path(data).read_text()),
            started=started, finished=time.monotonic())))

    def member_complete(self, member_dir):
        return (Path(member_dir) / "trained.json").is_file()


def trained(member_dir):
    return json.loads((Path(member_dir) / "trained.json").read_text())
