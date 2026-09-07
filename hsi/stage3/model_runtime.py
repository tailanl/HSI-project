"""Explicit external ReMoGen assets; no historical workspace discovery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from hsi.common.artifacts import artifact, require, verified


CHECKPOINT_HASHES = {
    "adapter_checkpoint": "519de22df9d7f3e2451c80e3029bbf3cba6cd86c0d54ae972fcb2291f75b7a99",
    "base_checkpoint": "373fcc0682d095e67d083f5976b014c8562732f6e104f797273a656d5a10089e",
    "mvae_checkpoint": "e1c103cc9d5a4916adb3261bdf79e69d599a3aa4960b669b4969c1c786a58773",
    "clip_checkpoint": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}


@dataclass(frozen=True)
class MotionRuntime:
    remogen_root: Path
    adapter_checkpoint: Path
    base_checkpoint: Path
    mvae_checkpoint: Path
    primitive_config: Path
    statistics_directory: Path
    scene_directory: Path
    clip_checkpoint: Path

    @classmethod
    def from_mapping(cls, value):
        require(set(value) == set(cls.__dataclass_fields__), "Exact ReMoGen runtime paths are required")
        return cls(**{key: Path(path).resolve(strict=True) for key, path in value.items()})

    def bind(self):
        required = {name: getattr(self, name) for name in CHECKPOINT_HASHES}
        required.update(primitive_config=self.primitive_config,
            base_args=self.base_checkpoint.parent / "args.yaml",
            mvae_args=self.mvae_checkpoint.parent / "args.yaml",
            normalization_statistics=self.statistics_directory / "mean_std_h2_f8.pkl",
            text_embedding_cache=self.statistics_directory / "val_text_embedding_dict.pkl")
        records = {key: artifact(path) for key, path in required.items()}
        for key, expected in CHECKPOINT_HASHES.items():
            require(records[key]["sha256"] == expected, "Current motion checkpoint mismatch: " + key)
        for directory in ("model", "mld", "data_loaders", "utils", "diffusion", "config_files"):
            root = self.remogen_root / directory
            require(root.is_dir(), "External ReMoGen implementation is missing " + directory)
            for path in sorted(root.rglob("*.py")):
                records["source:" + str(path.relative_to(self.remogen_root))] = artifact(path)
        import importlib.util
        spec = importlib.util.find_spec('clip')
        require(spec is not None and spec.origin, 'External OpenAI CLIP package is required')
        clip_root = Path(spec.origin).parent
        for name in ('clip.py', 'model.py', 'simple_tokenizer.py', '__init__.py', 'bpe_simple_vocab_16e6.txt.gz'):
            records['clip_source:' + name] = artifact(clip_root / name)
        return records


_runtime: MotionRuntime | None = None


def configure(runtime: MotionRuntime):
    global _runtime
    require(type(runtime) is MotionRuntime, "Use explicit MotionRuntime assets")
    root = runtime.remogen_root.resolve(strict=True)
    module = sys.modules.get("model")
    if module is not None and getattr(module, "__file__", None):
        require(Path(module.__file__).resolve().is_relative_to(root), "A different model package is already imported")
    _runtime = runtime
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def current():
    require(_runtime is not None, "ReMoGen runtime is not configured; no model paths are inferred")
    return _runtime
