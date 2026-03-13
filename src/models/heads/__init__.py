'''
Task-specific head architectures.
'''
from .classification import ClassificationHeadV1, LegacyIDFraudHead

HEAD_MAP = {
    'classification': {
        'v1': ClassificationHeadV1,
        'legacy': LegacyIDFraudHead,
    },
}


def build_head(task: str, head_type: str, input_dim: int):
    """Create head for a given task.
    Args:
        task: Task type, currently only supports 'classification'
        head_type: Head variant, currently classifcation task supports 'v1' and 'legacy' head
        input_dim: Input dimension (the output dimension from backbone)
    """
    if task not in HEAD_MAP:
        raise ValueError(f"Unknown task '{task}'. Available: {list(HEAD_MAP.keys())}")
    task_heads = HEAD_MAP[task]
    if head_type not in task_heads:
        raise ValueError(f"Unknown head_type '{head_type}' for task '{task}'. Available: {list(task_heads.keys())}")
    return task_heads[head_type](input_dim)
