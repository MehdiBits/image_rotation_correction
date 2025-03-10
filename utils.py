import torch

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # map_location argument allows to switch between cpu-gpu inference models
    checkpoint = torch.load(filename, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded: {filename}")
