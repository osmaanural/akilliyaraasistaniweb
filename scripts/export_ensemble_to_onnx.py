from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet, mobilenet_v2, resnet50, vit_b_16


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT.parent / "pressure_ulcer_project" / "saved_models"
DEFAULT_OUT = PROJECT_ROOT / "public" / "models"
CLASS_NAMES = ["Evre 1", "Evre 2", "Evre 3", "Evre 4"]
ENSEMBLE_WEIGHTS = {
    "resnet": 0.3,
    "mobilenet": 0.3,
    "alexnet": 0.1,
    "vit": 0.3,
}


def resnet_model() -> nn.Module:
    model = resnet50(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 4),
    )
    return model


def mobilenet_model() -> nn.Module:
    model = mobilenet_v2(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_feats, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4),
    )
    return model


def alexnet_model() -> nn.Module:
    model = alexnet(weights=None)
    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_feats, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 4),
    )
    return model


def vit_model() -> nn.Module:
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(768, 4)
    return model


def load_state_dict(model: nn.Module, path: Path) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_model(name: str, model: nn.Module, out_path: Path, verify: bool) -> dict:
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        torch_logits = model(dummy).detach().cpu().numpy()

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    max_abs_diff = None
    if verify:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        ort_logits = session.run(["logits"], {"input": dummy.numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))

    return {
        "name": name,
        "path": str(out_path.relative_to(PROJECT_ROOT)),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
        "max_abs_diff": max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AYA ensemble PyTorch modellerini ONNX'e çevirir.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", choices=["all", "mobilenet", "resnet", "alexnet", "vit"], default="all")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    specs = {
        "resnet": (resnet_model, args.source / "resnet_son.pth"),
        "mobilenet": (mobilenet_model, args.source / "mobilenet.pth"),
        "alexnet": (alexnet_model, args.source / "alexnet.pth"),
        "vit": (vit_model, args.source / "vitson_model.pth"),
    }
    selected = specs if args.only == "all" else {args.only: specs[args.only]}

    results = []
    for name, (factory, checkpoint_path) in selected.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
        print(f"{name}: yükleniyor -> {checkpoint_path}")
        model = load_state_dict(factory(), checkpoint_path)
        out_path = args.out / f"{name}.onnx"
        print(f"{name}: export -> {out_path}")
        results.append(export_model(name, model, out_path, verify=not args.no_verify))

    manifest = {
        "input": {
            "shape": [1, 3, 224, 224],
            "format": "RGB",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "classes": CLASS_NAMES,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "models": results,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
