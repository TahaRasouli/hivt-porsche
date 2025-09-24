import argparse
import torch
from torch.utils.data import DataLoader
from datasets import ArgoverseV1Dataset
from models import HiVT  # your main model that uses LocalEncoder + GlobalInteractor

# --- Safe checkpoint loader for PyTorch 2.6+ ---
from torch.serialization import safe_globals

def load_checkpoint(ckpt_path, map_location='cpu'):
    try:
        # allowlist the PyTorch Lightning ModelCheckpoint class
        with safe_globals([torch.nn.Module]):
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def evaluate(model, dataloader, device):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            local_embed = model.local_encoder(batch)
            global_embed = model.global_interactor(batch, local_embed)
            all_outputs.append(global_embed.cpu())
    return torch.cat(all_outputs, dim=1)  # [F, N, D]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Dataset root path')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load dataset
    dataset = HiVTDataset(args.root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # Initialize model
    model = HiVT()
    model.to(device)

    # Load checkpoint
    checkpoint = load_checkpoint(args.ckpt_path, map_location=device)
    if checkpoint is not None:
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        # Load non-strict to ignore missing/unexpected keys
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully (non-strict).")

    # Run evaluation
    outputs = evaluate(model, dataloader, device)
    print("Evaluation done. Output shape:", outputs.shape)

if __name__ == '__main__':
    main()
