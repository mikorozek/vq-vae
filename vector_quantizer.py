import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, scale=1.0, dtype=torch.float32):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost
        scale /= max(1., embedding_dim)
        limit = math.sqrt(3.0 * scale)
        with torch.no_grad():
            self.embeddings.weight.uniform_(-limit, limit)

    def __call__(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        flat_inputs = inputs.view(-1, self.embedding_dim)

        flat_inputs_normalized = F.normalize(flat_inputs, p=2.0, dim=1)
        embeddings_normalized = F.normalize(self.embeddings.weight, p=2.0, dim=1)

        similarity = torch.matmul(flat_inputs_normalized, embeddings_normalized.t())
        encoding_indices = torch.argmax(similarity, dim=1)

        quantized = self.quantize(encoding_indices)

        quantized = quantized.view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1)

        return quantized, loss, encoding_indices

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)








