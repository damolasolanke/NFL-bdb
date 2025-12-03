"""
Collate function for batching NFL trajectory graphs.

DeepMind-caliber batching with strict shape contracts and zero PyG dependencies.
"""

import torch
from typing import List, Dict


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching graphs with different numbers of nodes.
    
    Args:
        batch: List of samples, each with:
            - node_feats: [N, F]
            - edge_index: [2, E]
            - edge_attr: [E, 4]
            - initial_pos: [num_players, T, 2]
            - target: [num_players, T-1, 2]
            - mask: [num_players, T-1]
            - key: Tuple
            - trajectory_length: int
    
    Returns:
        Batched dictionary with:
            - node_feats: [B*N, F] (concatenated, with batch_index)
            - edge_index: [2, total_edges] (offset indices)
            - edge_attr: [total_edges, 4] (concatenated)
            - batch_index: [B*N] (batch assignment for each node)
            - initial_pos: [B, num_players, T, 2] (stacked)
            - target: [B, num_players, T-1, 2] (stacked)
            - mask: [B, num_players, T-1] (stacked)
            - keys: List[Tuple]
            - trajectory_lengths: [B] (tensor)
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")
    
    B = len(batch)
    
    # ====================================================================
    # 1. Extract and validate shapes from first sample
    # ====================================================================
    first_sample = batch[0]
    N = first_sample["node_feats"].shape[0]  # max_players (same for all)
    F = first_sample["node_feats"].shape[1]  # feature dim (same for all)
    num_players = first_sample["initial_pos"].shape[0]  # num_players (same for all)
    T = first_sample["initial_pos"].shape[1]  # max_trajectory_length (same for all)
    
    # ====================================================================
    # 2. Batch node features and create batch_index
    # ====================================================================
    node_feats_list = []
    batch_index_list = []
    
    for b_idx, sample in enumerate(batch):
        node_feats = sample["node_feats"]  # [N, F]
        assert node_feats.shape == (N, F), \
            f"Sample {b_idx}: node_feats shape must be ({N}, {F}), got {node_feats.shape}"
        
        node_feats_list.append(node_feats)
        batch_index_list.append(torch.full((N,), b_idx, dtype=torch.long))
    
    batched_node_feats = torch.cat(node_feats_list, dim=0)  # [B*N, F]
    batch_index = torch.cat(batch_index_list, dim=0)  # [B*N]
    
    # ====================================================================
    # 3. Batch edge indices and attributes (with offset)
    # ====================================================================
    edge_index_list = []
    edge_attr_list = []
    
    node_offset = 0
    for b_idx, sample in enumerate(batch):
        edge_index = sample["edge_index"]  # [2, E]
        edge_attr = sample["edge_attr"]  # [E, 4]
        
        assert edge_index.shape[0] == 2, \
            f"Sample {b_idx}: edge_index shape[0] must be 2, got {edge_index.shape[0]}"
        assert edge_attr.shape[1] == 4, \
            f"Sample {b_idx}: edge_attr shape[1] must be 4, got {edge_attr.shape[1]}"
        assert edge_index.shape[1] == edge_attr.shape[0], \
            f"Sample {b_idx}: edge_index and edge_attr must have matching edge count"
        
        # Offset edge indices by node_offset
        offset_edge_index = edge_index + node_offset
        edge_index_list.append(offset_edge_index)
        edge_attr_list.append(edge_attr)
        
        node_offset += N
    
    batched_edge_index = torch.cat(edge_index_list, dim=1)  # [2, total_edges]
    batched_edge_attr = torch.cat(edge_attr_list, dim=0)  # [total_edges, 4]
    
    # ====================================================================
    # 4. Batch temporal sequences (initial_pos, target, mask)
    # ====================================================================
    initial_pos_list = []
    target_list = []
    mask_list = []
    keys_list = []
    trajectory_lengths_list = []
    
    for sample in batch:
        initial_pos = sample["initial_pos"]  # [num_players, T, 2]
        target = sample["target"]  # [num_players, T-1, 2]
        mask = sample["mask"]  # [num_players, T-1]
        key = sample["key"]
        traj_len = sample["trajectory_length"]
        
        assert initial_pos.shape == (num_players, T, 2), \
            f"initial_pos shape must be ({num_players}, {T}, 2), got {initial_pos.shape}"
        assert target.shape == (num_players, T - 1, 2), \
            f"target shape must be ({num_players}, {T - 1}, 2), got {target.shape}"
        assert mask.shape == (num_players, T - 1,), \
            f"mask shape must be ({num_players}, {T - 1},), got {mask.shape}"
        
        initial_pos_list.append(initial_pos)
        target_list.append(target)
        mask_list.append(mask)
        keys_list.append(key)
        trajectory_lengths_list.append(traj_len)
    
    batched_initial_pos = torch.stack(initial_pos_list, dim=0)  # [B, num_players, T, 2]
    batched_target = torch.stack(target_list, dim=0)  # [B, num_players, T-1, 2]
    batched_mask = torch.stack(mask_list, dim=0)  # [B, num_players, T-1]
    trajectory_lengths = torch.tensor(trajectory_lengths_list, dtype=torch.long)  # [B]
    
    # ====================================================================
    # 5. Final shape validation
    # ====================================================================
    assert batched_node_feats.shape == (B * N, F), \
        f"batched_node_feats shape must be ({B * N}, {F}), got {batched_node_feats.shape}"
    assert batch_index.shape == (B * N,), \
        f"batch_index shape must be ({B * N},), got {batch_index.shape}"
    assert batched_edge_index.shape[0] == 2, \
        f"batched_edge_index shape[0] must be 2, got {batched_edge_index.shape[0]}"
    assert batched_edge_attr.shape[1] == 4, \
        f"batched_edge_attr shape[1] must be 4, got {batched_edge_attr.shape[1]}"
    assert batched_initial_pos.shape == (B, num_players, T, 2), \
        f"batched_initial_pos shape must be ({B}, {num_players}, {T}, 2), got {batched_initial_pos.shape}"
    assert batched_target.shape == (B, num_players, T - 1, 2), \
        f"batched_target shape must be ({B}, {num_players}, {T - 1}, 2), got {batched_target.shape}"
    assert batched_mask.shape == (B, num_players, T - 1), \
        f"batched_mask shape must be ({B}, {num_players}, {T - 1}), got {batched_mask.shape}"
    
    # ====================================================================
    # 6. Return batched dictionary
    # ====================================================================
    return {
        "node_feats": batched_node_feats.contiguous(),  # [B*N, F]
        "edge_index": batched_edge_index.contiguous(),  # [2, total_edges]
        "edge_attr": batched_edge_attr.contiguous(),  # [total_edges, 4]
        "batch_index": batch_index.contiguous(),  # [B*N]
        "initial_pos": batched_initial_pos.contiguous(),  # [B, T, 2]
        "target": batched_target.contiguous(),  # [B, T-1, 2]
        "mask": batched_mask.contiguous(),  # [B, T-1]
        "keys": keys_list,  # List[Tuple]
        "trajectory_lengths": trajectory_lengths  # [B]
    }




