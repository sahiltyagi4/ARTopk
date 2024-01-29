from flexible_compression.miscellaneous import Memory

class ResidualMemory(Memory):
    def __init__(self):
        self.residuals = {}
        self.layer_decompress = {}

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.residuals[name] + tensor

        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        self.layer_decompress[name] = tensor_decompressed
        residual = tensor - tensor_decompressed
        self.residuals[name] = residual