import wandb


def log_artifact(dir_path: str, job_type: str):
    run = wandb.init(project="fairness-autosklearn", job_type=job_type, name=job_type)
    files = wandb.Artifact(job_type, type="files")
    files.add_dir(dir_path)
    run.log_artifact(files)
    wandb.finish()
