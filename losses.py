import torch
import torch.nn.functional as F


def degree_error_distance(source_axis, target_axis):
    cosine_sim = F.cosine_similarity(source_axis, target_axis)
    eps = 1e-7
    return torch.acos(torch.clamp(cosine_sim, min=-1 + eps, max=1 - eps))


def cosine_similarity_distance(source_axis, target_axis):
    cosine_sim = F.cosine_similarity(source_axis, target_axis)
    return 1 - cosine_sim


def origin_error_distance(source_axis, source_v, target_axis, target_v):
    degree_difference = torch.acos(F.cosine_similarity(source_axis, target_axis))

    if degree_difference == 0:
        distance = torch.linalg.norm(torch.cross(target_axis, target_v - source_v))
    else:
        distance = torch.tensordot(target_axis, source_v) + torch.tensordot(source_axis, target_v) \
                   / torch.linalg.norm(torch.cross(target_axis, source_axis))

    return torch.abs(distance)
