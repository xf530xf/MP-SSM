import argparse
import json

from .config import load_config


def execution_flags(parser):
    parser.add_argument("--device", help="cuda[:index], cpu or mps")
    parser.add_argument("--scan-backend", choices=["cuda", "reference"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="MP-SSM paper-guided binary segmentation")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train with paired masks")
    training.add_argument("--config", default="configs/paper.yaml")
    training.add_argument("--data", help="Dataset root")
    training.add_argument("--output", required=True)
    training.add_argument("--resume", help="Full latest.pt checkpoint")
    training.add_argument("--epochs", type=int, help="Total target epochs, not extra epochs")
    execution_flags(training)
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved checkpoint")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--data")
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--output", required=True, help="JSON file")
    execution_flags(evaluation)
    inference = commands.add_parser("infer", help="Infer an image, image directory or video")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--source", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--native", action="store_true", help="Use original input resolution instead of training size")
    inference.add_argument("--threshold", type=float)
    inference.add_argument("--min-area", type=int, default=1)
    inference.add_argument("--max-frames", type=int)
    inference.add_argument("--save-every", type=int, default=1, help="Save video probability/mask every N frames; infer all frames")
    execution_flags(inference)
    performance = commands.add_parser("benchmark", help="Measure model-only performance")
    performance.add_argument("--config", default="configs/paper.yaml")
    performance.add_argument("--checkpoint")
    performance.add_argument("--output", required=True)
    performance.add_argument("--sizes", type=int, nargs="+", default=[512, 2048])
    performance.add_argument("--warmup", type=int, default=10)
    performance.add_argument("--iterations", type=int, default=50)
    performance.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    performance.add_argument("--allow-slow-reference", action="store_true")
    performance.add_argument("--profile-only", action="store_true", help="Count parameters and storage without a forward pass")
    execution_flags(performance)
    synthetic = commands.add_parser("synthetic", help="Create explicitly synthetic sample pairs")
    synthetic.add_argument("--output", default="data/synthetic")
    synthetic.add_argument("--count", type=int, default=4, help="Number of training images")
    synthetic.add_argument("--size", type=int, default=64)
    synthetic.add_argument("--seed", type=int, default=42)
    split = commands.add_parser("split", help="Split by complete source videos")
    split.add_argument("--source", required=True)
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command in {"train", "benchmark"}:
            config = load_config(args.config)
            if args.device:
                config["train"]["device"] = args.device
            if args.scan_backend:
                config["model"]["scan_backend"] = args.scan_backend
        if args.command == "train":
            from .engine import train
            if args.data:
                config["data"]["root"] = args.data
            if args.epochs is not None:
                config["train"]["epochs"] = args.epochs
            result = train(config, args.output, args.resume)
        elif args.command == "evaluate":
            from .engine import evaluate
            result = evaluate(args.checkpoint, args.data, args.output, args.split, args.device, args.scan_backend)
        elif args.command == "infer":
            from .inference import infer
            result = infer(args.checkpoint, args.source, args.output, args.device, args.scan_backend,
                           args.native, args.threshold, args.min_area, args.max_frames, args.save_every)
        elif args.command == "benchmark":
            from .benchmark import benchmark
            result = benchmark(config, args.output, args.sizes, args.warmup, args.iterations, args.precision,
                               args.checkpoint, args.allow_slow_reference, args.profile_only)
        elif args.command == "synthetic":
            from .data import make_synthetic
            result = make_synthetic(args.output, args.count, args.size, args.seed)
        elif args.command == "split":
            from .data import split_by_video
            result = split_by_video(args.source, args.manifest, args.output, args.seed)
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        parser.exit(1, f"MP-SSM error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
