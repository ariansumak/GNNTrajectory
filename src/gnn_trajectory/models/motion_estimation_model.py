import torch
import torch.nn as nn
from pathlib import Path

from gnn_trajectory.models.decoder import MotionDecoder
from gnn_trajectory.models.decoder_mlp import MLPDisplacementDecoder
from gnn_trajectory.models.encoder_gat import MotionEncoder
from gnn_trajectory.models.encoder_gcn import MotionEncoderGCN

from gnn_trajectory.data.argoverse2_dataset import AV2GNNForecastingDataset

class MotionForecastModel(nn.Module):
    def __init__(self, encoder_cfg=None, decoder_cfg=None):
        super().__init__()
        #self.encoder = MotionEncoder(**(encoder_cfg or {}))
        self.encoder = MotionEncoderGCN(**(encoder_cfg or {}))
        #self.decoder = MotionDecoder(**(decoder_cfg or {}))
        self.decoder = MLPDisplacementDecoder(**(decoder_cfg or {}))
        

    def forward(self, batch):
        # Encode the scene
        enc_out = self.encoder(batch)
        agent_map = enc_out["agent_map"]
        agent_pos_T = batch["agent_pos_T"]

        # Find focal agent (closest to origin)
        dists = torch.norm(agent_pos_T, dim=1)
        focal_idx = torch.argmin(dists).item()
        focal_feat = agent_map[focal_idx].unsqueeze(0)
        start_pos = agent_pos_T[focal_idx].unsqueeze(0)

        # Decode trajectory
        pred_traj = self.decoder(focal_feat, start_pos=start_pos)
        return pred_traj, focal_idx


# ------------------------------------------------------------
# Main: test run
# ------------------------------------------------------------
def main():
    root = Path("/home/silviu/Documents/Workspace/DeepLearning/Project/av2-api")
    dataset = AV2GNNForecastingDataset(root=root, split="val")
    sample1 = dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    sample2 = next(iter(dataloader))
    print(f"Loaded scenario: {sample['scenario_id']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample.items()}

    model = MotionForecastModel().to(device)
    model.eval()

    with torch.no_grad():
        pred_traj, focal_idx = model(sample)

    print(f"Predicted trajectory shape: {pred_traj.shape}")
    print(f"Focal agent index: {focal_idx}")
    print(f"First few predicted coords:\n{pred_traj[0, :5].cpu().numpy()}")

if __name__ == "__main__":
    main()
