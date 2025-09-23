"""
Warmup script: loads models and runs one dummy forward pass to avoid cold-start delay.
"""
from app.deps import get_config, get_models

if __name__ == "__main__":
    cfg = get_config()
    models = get_models()
    print("Models warmed up successfully.")
