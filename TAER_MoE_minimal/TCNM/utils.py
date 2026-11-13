import torch

def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)




def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


def get_dset_path(dset_name, dset_type):
    return {'root':dset_name,'type':dset_type}


def relative_to_abs(rel_traj, start_pos, feature6_data=None, feature10_data=None):
    """
    Convert relative trajectory to absolute trajectory.
    
    Args:
        rel_traj (tensor): Relative trajectory (seq_len, batch, 2)
        start_pos (tensor): Start position (batch, 2)
        feature6_data (tensor, optional): Additional feature6 data (default: None)
        feature10_data (tensor, optional): Additional feature10 data (default: None)
    
    Returns:
        abs_traj (tensor): Absolute trajectory (seq_len, batch, 2)
    """
    # Compute displacement from relative trajectory
    rel_traj = rel_traj.permute(2, 1, 0, 3)
    displacement = torch.cumsum(rel_traj, dim=2)

    # Add start position
    abs_traj = displacement + start_pos[:, None, None]

    # Optionally include feature6 and feature10 if available
    if feature6_data is not None:
        abs_traj += feature6_data
    if feature10_data is not None:
        abs_traj += feature10_data

    return abs_traj.permute(2, 1, 0, 3)


