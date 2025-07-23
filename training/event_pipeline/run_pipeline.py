from training.event_pipeline.pipeline import TrainPipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    pipeline = TrainPipeline(config_path=args.config)
    pipeline()
