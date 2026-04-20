import os
import pickle
import torch
import torch.nn as nn

INPUT_DIR  = 'outputs/v2x_cooperative'
OUTPUT_DIR = 'outputs/v2x_cooperative'

os.makedirs(OUTPUT_DIR, exist_ok=True)


class BeliefEncoder(nn.Module):
    """
    Encodes agent state into a compressed latent belief vector.
    This is the key PS-2 contribution —
    transmit beliefs not raw features.
    """
    def __init__(self, input_dim=6, latent_dim=128,
                 compressed_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        self.compressor = nn.Sequential(
            nn.Linear(latent_dim, compressed_dim),
            nn.Tanh()
        )
        self.decompressor = nn.Sequential(
            nn.Linear(compressed_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim)
        )

    def encode(self, x):
        latent     = self.encoder(x)
        compressed = self.compressor(latent)
        return compressed

    def decode(self, compressed):
        return self.decompressor(compressed)

    def forward(self, x):
        compressed = self.encode(x)
        recovered  = self.decode(compressed)
        return compressed, recovered


agent_path = os.path.join(INPUT_DIR, 'agent_states.pkl')

if not os.path.exists(agent_path):
    print("Agent states not found.")
    print("Run 01_multi_agent_setup.py first.")
    exit()

with open(agent_path, 'rb') as f:
    agent_states = pickle.load(f)

model     = BeliefEncoder(input_dim=6, latent_dim=128, compressed_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

state_vectors = []
for state in agent_states.values():
    pos = state['position']
    vel = state['velocity']
    vec = pos + vel
    state_vectors.append(vec)

X = torch.tensor(state_vectors, dtype=torch.float32)

print(f"Training belief encoder on {len(X)} agents...")
print()

EPOCHS = 50
losses = []
model.train()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    compressed, recovered = model(X)
    loss = criterion(recovered, X)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:>3}/{EPOCHS}  "
              f"Loss: {loss.item():.6f}  "
              f"Compression ratio: {16/6:.2f}x")

model.eval()
with torch.no_grad():
    compressed_beliefs, _ = model(X)

print()
print(f"Original state dim    : {X.shape[1]}")
print(f"Compressed belief dim : {compressed_beliefs.shape[1]}")
print(f"Compression ratio     : {X.shape[1]/compressed_beliefs.shape[1]:.2f}x")
print(f"Compressed beliefs    : {compressed_beliefs.shape}")

out_path = os.path.join(OUTPUT_DIR, 'compressed_beliefs.pt')
torch.save(compressed_beliefs, out_path)
print(f"\nCompressed beliefs saved: {out_path}")