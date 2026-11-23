"""Train PriorFuse and report patient-level out-of-fold performance."""

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import random
import re
import tempfile

import numpy as np
from PIL import Image, ImageStat
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as image_ops

from backbone import load_clip_backbone
from medical_priors import (
    FrozenMedicalPriors, MedicalPriorAlignment, MODALITIES, SLICE_COUNTS,
    difference_alignment_loss,
)
from opts import parse_args


FOLD_COUNT = 5
INNER_VALIDATION_FRACTION = 0.2
MIXUP_ALPHA = 0.2
ALIGNMENT_WEIGHT = 1.0
IMAGE_SIZE = 224
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
BILINEAR = InterpolationMode.BILINEAR


@dataclass(frozen=True)
class Scan:
    paths: tuple[Path, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class Patient:
    patient_id: str
    label: int
    scans: dict[str, Scan]


@dataclass(frozen=True)
class Fold:
    number: int
    train: tuple[Patient, ...]
    validation: tuple[Patient, ...]
    outer: tuple[Patient, ...]


def natural_key(value):
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"([0-9]+)", value.casefold()))


def read_slice(path):
    with Image.open(path) as image:
        return image_ops.resize(image.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), BILINEAR)


def read_scan(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing modality directory: {folder}")
    paths = tuple(sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: (natural_key(p.name), p.name),
    ))
    if not paths:
        raise ValueError(f"No supported slices found in {folder}")
    scores = []
    for path in paths:
        try:
            scores.append(ImageStat.Stat(read_slice(path).convert("L")).stddev[0])
        except OSError as exc:
            raise ValueError(f"Cannot read slice {path}") from exc
    return Scan(paths, tuple(scores))


def read_cohort(data_root):
    patients = {}
    for partition in ("train", "test"):
        label_path = data_root / "label" / f"{partition}_label.csv"
        with label_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not {"patient_id", "label"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"Expected patient_id and label columns in {label_path}")
            for row in reader:
                patient_id = (row["patient_id"] or "").strip()
                label = (row["label"] or "").strip()
                if not patient_id or label not in {"0", "1"}:
                    raise ValueError(f"Invalid patient ID or binary label at {label_path}:{reader.line_num}")
                if patient_id in patients:
                    raise ValueError(f"Duplicate patient ID: {patient_id}")
                scans = {
                    name: read_scan(data_root / partition / name / patient_id) for name in MODALITIES
                }
                patients[patient_id] = Patient(patient_id, int(label), scans)
    return sorted(patients.values(), key=lambda patient: (natural_key(patient.patient_id), patient.patient_id))


def make_folds(patients, seed):
    labels = np.array([patient.label for patient in patients], dtype=int)
    if np.bincount(labels, minlength=2).min() < FOLD_COUNT:
        raise ValueError(f"Each class needs at least {FOLD_COUNT} patients for stratification")
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=seed)

    def select(indices):
        return tuple(patients[index] for index in sorted(indices))

    for number, (development, outer) in enumerate(splitter.split(labels, labels), start=1):
        try:
            train, validation = train_test_split(
                development, test_size=INNER_VALIDATION_FRACTION,
                stratify=labels[development], random_state=seed + number,
            )
        except ValueError as exc:
            raise ValueError(f"Cannot stratify the inner validation set for fold {number}: {exc}") from exc
        if np.unique(labels[train]).size != 2 or np.unique(labels[validation]).size != 2:
            raise ValueError(f"Both classes are required in training and validation for fold {number}")
        yield Fold(number, select(train), select(validation), select(outer))


def select_slices(scan, count, training):
    start, stop = 0, len(scan.paths)
    if training and stop > 2 * count:
        window = 2 * count
        peak = max(range(stop), key=scan.scores.__getitem__)
        start = random.randint(max(0, peak - window + 1), min(peak, stop - window))
        stop = start + window
    ranked = sorted(range(start, stop), key=lambda index: (-scan.scores[index], index))
    selected = [scan.paths[index] for index in sorted(ranked[:count])]
    return selected + [selected[-1]] * (count - len(selected))


def augment_slices(images):
    """Apply the same geometric parameters to every slice of one sequence."""
    flip = random.random() < 0.5
    rotation = random.uniform(-10, 10)
    translation = [round(random.uniform(-0.05, 0.05) * IMAGE_SIZE) for _ in range(2)]
    scale = random.uniform(0.95, 1.05)

    if flip:
        images = [image_ops.hflip(image) for image in images]
    images = [image_ops.rotate(image, rotation, BILINEAR) for image in images]
    images = [
        image_ops.affine(image, 0, translation, scale, [0, 0], interpolation=BILINEAR)
        for image in images
    ]

    # Geometric RandAugment, N=2 and M=7 on torchvision's 30-step magnitude scale.
    for _ in range(2):
        operation = random.choice(("rotate", "translate_x", "translate_y", "shear_x", "shear_y"))
        magnitude = random.choice((-1, 1)) * 7 / 30
        if operation == "rotate":
            images = [image_ops.rotate(image, 30 * magnitude, BILINEAR) for image in images]
        elif operation.startswith("translate"):
            shift = int((150 / 331) * IMAGE_SIZE * magnitude)
            translation = [shift, 0] if operation == "translate_x" else [0, shift]
            images = [
                image_ops.affine(image, 0, translation, 1, [0, 0], interpolation=BILINEAR)
                for image in images
            ]
        else:
            shear_angle = float(np.degrees(np.arctan(0.3 * magnitude)))
            shear = [shear_angle, 0] if operation == "shear_x" else [0, shear_angle]
            images = [
                image_ops.affine(image, 0, [0, 0], 1, shear, interpolation=BILINEAR, center=[0, 0])
                for image in images
            ]
    return images


class PatientDataset(Dataset):
    def __init__(self, patients, training=False):
        self.patients = patients
        self.training = training

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        patient = self.patients[index]
        modalities = {}
        for name in MODALITIES:
            paths = select_slices(patient.scans[name], SLICE_COUNTS[name], self.training)
            by_path = {path: read_slice(path) for path in dict.fromkeys(paths)}
            images = [by_path[path] for path in paths]
            if self.training:
                images = augment_slices(images)
            modalities[name] = torch.stack([image_ops.to_tensor(image) for image in images])
        return modalities, patient.label, patient.patient_id


class PriorFuse(nn.Module):
    """Predict END from the shared ViT features of the three modalities."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        width = backbone.width
        self.classifier = nn.Sequential(
            nn.Linear(width * len(MODALITIES), 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def encode(self, images):
        features, contexts = [], {}
        for name in MODALITIES:
            feature, contexts[name] = self.backbone(images[name])
            features.append(feature)
        return torch.stack(features, dim=1), contexts

    def classify(self, shared):
        return self.classifier(shared.flatten(1))

    def forward(self, images):
        shared, _ = self.encode(images)
        return self.classify(shared)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(patients, args, device, seed, training=False):
    return DataLoader(
        PatientDataset(patients, training), batch_size=args.batch_size, shuffle=training,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed), persistent_workers=args.num_workers > 0,
    )


def prepare_batch(images, labels, device):
    images = {name: value.to(device, non_blocking=True) for name, value in images.items()}
    return images, labels.to(device, non_blocking=True)


def mix_batch(images, cta_pixels, labels):
    ratio = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    permutation = torch.randperm(len(labels), device=labels.device)
    targets = F.one_hot(labels, num_classes=2).float()
    images = {name: value.lerp(value[permutation], 1 - ratio) for name, value in images.items()}
    return images, cta_pixels.lerp(cta_pixels[permutation], 1 - ratio), targets.lerp(
        targets[permutation], 1 - ratio
    )


def train_epoch(model, priors, mpam, loader, optimizer, scaler, class_weights, args, device):
    model.train()
    mpam.train()
    total_loss = torch.zeros((), device=device)
    for images, labels, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        cta_pixels = priors.prepare_cta(images["CTA"]).to(device, non_blocking=True)
        images, labels = prepare_batch(images, labels, device)
        images, cta_pixels, targets = mix_batch(images, cta_pixels, labels)
        with torch.autocast(device.type, enabled=scaler.is_enabled(), dtype=torch.float16):
            shared, contexts = model.encode(images)
            logits = model.classify(shared)
            medical = mpam(contexts, priors(images, cta_pixels))
            # For the MixUp END target t, smoothing gives q = (1 - 2ε)t + ε.
            # The loss weights (1 - q, q) by class and divides by their total weighted mass.
            smoothed = targets * (1 - 2 * args.label_smoothing) + args.label_smoothing
            weighted_targets = smoothed * class_weights
            class_loss = -(weighted_targets * logits.float().log_softmax(dim=1)).sum()
            class_loss = class_loss / weighted_targets.sum()
            alignment = difference_alignment_loss(shared, medical, args.temperature)
            loss = class_loss + ALIGNMENT_WEIGHT * alignment
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach() * len(labels)
    average = total_loss.item() / len(loader.dataset)
    if not np.isfinite(average):
        raise RuntimeError("Training produced a non-finite loss")
    return average


def classification_metrics(rows):
    labels = [row["true_label"] for row in rows]
    predictions = [row["pred_label"] for row in rows]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    loss = torch.zeros((), device=device)
    for images, labels, ids in loader:
        images, labels = prepare_batch(images, labels, device)
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
            logits = model(images)
        loss += F.cross_entropy(logits.float(), labels, reduction="sum")
        probabilities = logits.float().softmax(dim=1)
        predictions = probabilities.argmax(dim=1).tolist()
        for patient_id, label, prediction, probability in zip(
            ids, labels.tolist(), predictions, probabilities[:, 1].tolist()
        ):
            rows.append({
                "patient_id": patient_id, "true_label": label,
                "pred_label": prediction, "prob_end": probability,
            })
    metrics = classification_metrics(rows)
    metrics["loss"] = loss.item() / len(rows)
    if not np.isfinite(metrics["loss"]):
        raise RuntimeError("Evaluation produced a non-finite loss")
    return metrics, rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        file.write("\n")


def save_checkpoint(path, model, fold, epoch, metrics):
    # Save the trainable prediction path. Frozen ViT weights come from the CLIP checkpoint.
    state = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
    temporary = path.with_suffix(".tmp")
    torch.save(
        {"fold": fold, "epoch": epoch, "validation": metrics, "trainable_state": state},
        temporary,
    )
    temporary.replace(path)


def restore_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    state = checkpoint["trainable_state"]
    if expected != state.keys():
        raise RuntimeError(f"Checkpoint does not match the trainable model parameters: {path}")
    model.load_state_dict(state, strict=False)
    return checkpoint


def run_fold(fold, priors, args, device, output_dir):
    fold_dir = output_dir / f"fold_{fold.number:02d}"
    fold_dir.mkdir()
    seed = args.seed + fold.number
    seed_everything(seed)
    backbone = load_clip_backbone(
        args.clip_weights, args.lora_rank, args.code_number, checkpoint_blocks=True
    )
    model = PriorFuse(backbone).to(device)
    mpam = MedicalPriorAlignment(backbone.width, priors.output_widths).to(device)
    optimizer = torch.optim.AdamW(
        (param for module in (model, mpam) for param in module.parameters() if param.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    positive = sum(patient.label for patient in fold.train)
    positive_weight = (len(fold.train) - positive) / positive
    class_weights = torch.tensor([1.0, positive_weight], device=device)
    train_loader = make_loader(fold.train, args, device, seed, training=True)
    validation_loader = make_loader(fold.validation, args, device, seed)
    checkpoint_path = fold_dir / "best.pt"

    history = []
    best_key = (-1.0, -float("inf"))
    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(
            model, priors, mpam, train_loader, optimizer, scaler, class_weights, args, device
        )
        validation, _ = evaluate(model, validation_loader, device)
        history.append({
            "epoch": epoch, "learning_rate": learning_rate, "train_loss": train_loss,
            **{f"validation_{key}": value for key, value in validation.items()},
        })
        candidate = (validation["accuracy"], -validation["loss"])
        if candidate > best_key:
            save_checkpoint(checkpoint_path, model, fold.number, epoch, validation)
            best_key = candidate
        print(f"Fold {fold.number}/{FOLD_COUNT} epoch {epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} validation_accuracy={validation['accuracy']:.4f}", flush=True)
        scheduler.step()
    write_csv(fold_dir / "history.csv", history)

    checkpoint = restore_checkpoint(checkpoint_path, model)
    outer_loader = make_loader(fold.outer, args, device, seed)
    outer_metrics, predictions = evaluate(model, outer_loader, device)
    print(f"Fold {fold.number} outer accuracy={outer_metrics['accuracy']:.4f} "
          f"using epoch {checkpoint['epoch']}", flush=True)
    record = {
        "fold": fold.number, "best_epoch": checkpoint["epoch"], "n_train": len(fold.train),
        "n_validation": len(fold.validation), "n_outer": len(fold.outer),
        "positive_class_weight": positive_weight,
        "validation_accuracy": checkpoint["validation"]["accuracy"],
        "validation_loss": checkpoint["validation"]["loss"],
        "checkpoint": checkpoint_path.relative_to(output_dir).as_posix(),
        **outer_metrics,
    }
    predictions = [{"fold": fold.number, **row} for row in predictions]
    write_csv(fold_dir / "predictions.csv", predictions)
    write_json(fold_dir / "metrics.json", record)
    return record, predictions


def pooled_metrics(rows):
    labels = [row["true_label"] for row in rows]
    scores = [row["prob_end"] for row in rows]
    return {
        **classification_metrics(rows),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
    }


def run_experiment(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unavailable device: {device}. Use CPU or install CUDA-enabled PyTorch.")

    patients = read_cohort(args.data_root)
    folds = tuple(make_folds(patients, args.seed))
    for key in ("data_root", "clip_weights", "brainiac_weights", "medclip_dir", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("priorfuse_%Y%m%d_%H%M%S_")
    output_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=args.output_dir))
    print(f"Running {FOLD_COUNT} folds for {len(patients)} patients on {device}. "
          f"Results: {output_dir}", flush=True)
    write_json(output_dir / "run_config.json", {
        **vars(args), "device": str(device), "fold_count": FOLD_COUNT,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION, "slice_counts": SLICE_COUNTS,
        "mixup_alpha": MIXUP_ALPHA, "alignment_weight": ALIGNMENT_WEIGHT,
        "medical_priors_usage": "Training alignment loss",
        "brainiac_input": "Sparse dADC and SB1000 slices, separately stacked and resampled to 96x96x96",
        "checkpoint_format": "Trainable prediction weights. Rebuild PriorFuse with the configured CLIP weights.",
    })
    write_csv(output_dir / "splits.csv", [
        {"fold": fold.number, "subset": subset, "patient_id": patient.patient_id, "label": patient.label}
        for fold in folds
        for subset, group in (("train", fold.train), ("validation", fold.validation), ("outer", fold.outer))
        for patient in group
    ])

    priors = FrozenMedicalPriors.from_pretrained(args.medclip_dir, args.brainiac_weights).to(device)
    fold_metrics, predictions = [], []
    for fold in folds:
        metrics, rows = run_fold(fold, priors, args, device, output_dir)
        fold_metrics.append(metrics)
        predictions.extend(rows)
    if sorted(row["patient_id"] for row in predictions) != sorted(p.patient_id for p in patients):
        raise RuntimeError("Expected exactly one out-of-fold prediction for each patient")

    best_fold = max(fold_metrics, key=lambda row: (row["accuracy"], -row["fold"]))
    write_csv(output_dir / "fold_metrics.csv", fold_metrics)
    write_csv(output_dir / "out_of_fold_predictions.csv", predictions)
    write_json(output_dir / "summary.json", {
        "n_patients": len(patients), "best_fold": best_fold, "pooled": pooled_metrics(predictions),
    })
    print(f"Best outer fold: {best_fold['fold']}, accuracy={best_fold['accuracy']:.4f}, "
          f"checkpoint={output_dir / best_fold['checkpoint']}", flush=True)
    return output_dir


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_experiment(parse_args())
