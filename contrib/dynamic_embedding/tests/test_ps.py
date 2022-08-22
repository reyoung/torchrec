import os
import unittest

import torch
from torchrec_dynamic_embedding import PS

torch.ops.tde.register_io(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_io/memory_io.so")
)


class TestPS(unittest.TestCase):
    def testEvictFetch(self):
        idx = torch.tensor([0, 2, 4, 8], dtype=torch.long)
        tensor = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        ps = PS("table", [tensor], "memory://")
        ps.evict(idx)
        tensor[:, :] = 0
        ps.fetch(idx)
        self.assertTrue(torch.allclose(tensor[idx], origin_tensor[idx]))

    def testOS(self):
        idx = torch.tensor([1, 3, 6], dtype=torch.long)
        tensor = torch.rand((10, 4))
        optim1 = torch.rand((10, 4))
        optim2 = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        origin_optim1 = optim1.clone()
        origin_optim2 = optim2.clone()
        ps = PS("table", [tensor, optim1, optim2], "memory://")
        ps.evict(idx)
        tensor[:, :] = 0
        optim1[:, :] = 0
        optim2[:, :] = 0
        ps.fetch(idx)
        self.assertTrue(torch.allclose(tensor[idx], origin_tensor[idx]))
        self.assertTrue(torch.allclose(optim1[idx], origin_optim1[idx]))
        self.assertTrue(torch.allclose(optim2[idx], origin_optim2[idx]))

    def testFetchNonExist(self):
        evict_idx = torch.tensor([0, 2, 4], dtype=torch.long)
        tensor = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        ps = PS("table", [tensor], "memory://")
        ps.evict(evict_idx)
        tensor[:, :] = 0
        additional_fetch_idx = torch.tensor([3, 9], dtype=torch.long)
        ps.fetch(torch.cat([evict_idx, additional_fetch_idx]))
        self.assertTrue(torch.allclose(tensor[evict_idx], origin_tensor[evict_idx]))
        self.assertTrue(
            torch.allclose(
                tensor[additional_fetch_idx],
                torch.zeros_like(tensor[additional_fetch_idx]),
            )
        )


if __name__ == "__main__":
    unittest.main()
